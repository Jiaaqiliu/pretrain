"""Detailed per-layer / per-expert / per-attention-projection spectral measurement.

Unlike the Phase 1/2 scripts (which aggregate experts and attention to global
means), this stores EVERY measured matrix individually so we can plot:
  - attention (q/k/v/o) vs FFN-expert heatmaps per layer
  - per-expert variation within each layer (which experts diverge)

Output: one JSON per model with a flat list of measured matrices, each tagged
with {layer, kind, proj, expert_id, alpha, srd, psi, entropy, shape}.
"""
import sys, os, time, json, gc
sys.path.insert(0, "/fsx/dev/jiaqi/moe_test")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/fsx/dev/jiaqi/.cache/huggingface"

import re
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from experiments.thermodynamics.moe_measures import (
    detect_architecture, MOE_ARCHITECTURES, measure_expert_weight,
    compute_svd, compute_alpha, compute_stable_rank,
    compute_order_parameter, compute_spectral_entropy,
)

OUTDIR = "/fsx/dev/jiaqi/moe_test/results/perlayer_detail"
os.makedirs(OUTDIR, exist_ok=True)


def measure_matrix(w, hidden_dim):
    with torch.no_grad():
        sv = compute_svd(w)
        return {
            "alpha": float(compute_alpha(sv)),
            "sr": float(compute_stable_rank(sv)),
            "srd": float(compute_stable_rank(sv) / hidden_dim) if hidden_dim else 0.0,
            "psi": float(compute_order_parameter(sv)),
            "entropy": float(compute_spectral_entropy(sv)),
            "shape": list(w.shape),
        }


def run(model_id, max_experts=0, max_layers=0, out_name=None):
    print(f"\n{'='*70}\nMODEL: {model_id}\n{'='*70}", flush=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hidden_dim = getattr(config, "hidden_size", 0) or getattr(config, "d_model", 0)
    print(f"[1] config: hidden={hidden_dim}", flush=True)

    print("[2] loading model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    print(f"    loaded in {time.time()-t0:.0f}s", flush=True)

    keys = list(model.state_dict().keys())
    arch = detect_architecture(keys)
    p = MOE_ARCHITECTURES[arch]
    print(f"[3] arch={arch}", flush=True)

    expert_re = re.compile(p["expert_pattern"]) if p["expert_pattern"] else None
    fused_re = re.compile(p["fused_expert_pattern"]) if p.get("fused_expert_pattern") else None
    router_re = re.compile(p["router_pattern"]) if p["router_pattern"] else None
    attn_re = re.compile(p["attn_pattern"]) if p["attn_pattern"] else None

    records = []  # flat list of measured matrices

    # First pass: discover layers + experts
    expert_store = {}   # {layer: {expert: {proj: tensor}}}
    attn_store = {}     # {layer: {proj: tensor}}
    router_store = {}   # {layer: tensor}

    for name, param in model.named_parameters():
        if param.ndim == 3 and fused_re:
            m = fused_re.match(name)
            if m:
                li, pt = int(m.group(1)), m.group(2)
                for ei in range(param.shape[0]):
                    expert_store.setdefault(li, {}).setdefault(ei, {})[pt] = param.data[ei]
                continue
        if param.ndim != 2:
            continue
        if expert_re:
            m = expert_re.match(name)
            if m:
                li, ei, pt = int(m.group(1)), int(m.group(2)), m.group(3)
                expert_store.setdefault(li, {}).setdefault(ei, {})[pt] = param.data
                continue
        if attn_re:
            m = attn_re.match(name)
            if m:
                li, pt = int(m.group(1)), m.group(2)
                attn_store.setdefault(li, {})[pt] = param.data
                continue
        if router_re:
            m = router_re.match(name)
            if m:
                router_store[int(m.group(1))] = param.data
                continue

    all_layers = sorted(expert_store.keys())
    if max_layers > 0 and len(all_layers) > max_layers:
        idx = np.linspace(0, len(all_layers) - 1, max_layers).astype(int)
        sel_layers = [all_layers[i] for i in idx]
    else:
        sel_layers = all_layers
    print(f"[4] measuring {len(sel_layers)} layers (of {len(all_layers)})", flush=True)

    t0 = time.time()
    for li in sel_layers:
        # Attention projections (each q/k/v/o separately)
        for pt, w in attn_store.get(li, {}).items():
            r = measure_matrix(w, hidden_dim)
            r.update({"layer": li, "kind": "attn", "proj": pt, "expert_id": -1})
            records.append(r)

        # Router
        if li in router_store:
            w = router_store[li]
            r = measure_matrix(w, w.shape[1])  # normalize by in-dim
            r.update({"layer": li, "kind": "router", "proj": "gate", "expert_id": -1})
            records.append(r)

        # Experts (each expert, each proj separately)
        experts = expert_store[li]
        eids = sorted(experts.keys())
        if max_experts > 0 and len(eids) > max_experts:
            idx = np.linspace(0, len(eids) - 1, max_experts).astype(int)
            eids = [eids[i] for i in idx]
        for ei in eids:
            for pt, w in experts[ei].items():
                r = measure_matrix(w, hidden_dim)
                r.update({"layer": li, "kind": "expert", "proj": pt, "expert_id": ei})
                records.append(r)

        n_attn = len(attn_store.get(li, {}))
        n_exp = len(eids)
        print(f"    layer {li:>2}: {n_attn} attn proj, {n_exp} experts measured", flush=True)

    print(f"[5] done in {time.time()-t0:.0f}s, {len(records)} matrices", flush=True)

    out = {
        "model": model_id,
        "arch": arch,
        "hidden_dim": hidden_dim,
        "intermediate_size": getattr(config, "intermediate_size", 0),
        "num_experts": (getattr(config, "num_local_experts", 0)
                        or getattr(config, "num_experts", 0)
                        or getattr(config, "n_routed_experts", 0) or 0),
        "layers_measured": sel_layers,
        "records": records,
    }
    fn = os.path.join(OUTDIR, out_name or (arch + "_detail.json"))
    with open(fn, "w") as f:
        json.dump(out, f)
    print(f"[6] saved {fn}", flush=True)

    del model
    gc.collect()
    return out


if __name__ == "__main__":
    print(f"START {time.strftime('%H:%M:%S UTC', time.gmtime())}", flush=True)
    # OLMoE: all 16 layers, sample 16 of 64 experts (enough for per-expert variation)
    run("allenai/OLMoE-1B-7B-0924", max_experts=16, max_layers=0,
        out_name="olmoe_detail.json")
    # Mixtral: all 8 experts, sample 8 layers
    run("mistralai/Mixtral-8x7B-v0.1", max_experts=8, max_layers=8,
        out_name="mixtral_detail.json")
    print(f"\nALL DONE {time.strftime('%H:%M:%S UTC', time.gmtime())}", flush=True)
