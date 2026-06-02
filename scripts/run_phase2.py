"""Phase 2: Cross-model spectral comparison.

Measure Mixtral-8x7B and Phi-3.5-MoE final checkpoints.
Key question: does per-expert intermediate_size determine alpha regime?

Expected:
- OLMoE: 64 experts, intermediate=1024, alpha~1.45 (already measured)
- Mixtral: 8 experts, intermediate=14336, alpha > 2? (large experts)
- Phi-3.5: 16 experts, intermediate=8192, alpha ~2? (medium experts)
"""
import sys, os, time, json, gc
sys.path.insert(0, "/fsx/dev/jiaqi/moe_test")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/fsx/dev/jiaqi/.cache/huggingface"

import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
from experiments.thermodynamics.moe_measures import (
    detect_architecture, MOE_ARCHITECTURES, measure_expert_weight,
    compute_svd, compute_cross_expert_alignment, compute_epr,
    compute_stable_rank, compute_order_parameter, compute_alpha,
    compute_spectral_entropy
)
import re

RESULTS_FILE = "/fsx/dev/jiaqi/moe_test/results/moe_cross_model/phase2_results.jsonl"


def measure_model(model_id, max_experts_per_layer=4, max_layers=4):
    """Measure a single model, sampling experts and layers for speed."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_id}")
    print(f"{'='*70}")

    print("[1] Loading config...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hidden_dim = getattr(config, "hidden_size", 0) or getattr(config, "d_model", 0)
    num_experts = (
        getattr(config, "num_local_experts", 0) or
        getattr(config, "num_experts", 0) or
        getattr(config, "n_routed_experts", 0) or 0
    )
    intermediate = getattr(config, "intermediate_size", 0)
    num_layers = getattr(config, "num_hidden_layers", 0)
    top_k = (
        getattr(config, "num_experts_per_tok", 0) or
        getattr(config, "num_experts_per_token", 0) or
        getattr(config, "top_k", 0) or 0
    )
    print(f"  hidden_dim={hidden_dim}, intermediate={intermediate}")
    print(f"  num_experts={num_experts}, top_k={top_k}, layers={num_layers}")

    print("[2] Loading model (fp16, CPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    t1 = time.time()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {t1-t0:.1f}s, params={total_params:,}")

    print("[3] Detecting architecture...")
    keys = list(model.state_dict().keys())
    arch = detect_architecture(keys)
    patterns = MOE_ARCHITECTURES[arch]
    print(f"  Architecture: {arch}")

    # Extract weights
    expert_re = re.compile(patterns["expert_pattern"]) if patterns["expert_pattern"] else None
    fused_re = re.compile(patterns["fused_expert_pattern"]) if patterns.get("fused_expert_pattern") else None
    router_re = re.compile(patterns["router_pattern"]) if patterns["router_pattern"] else None
    attn_re = re.compile(patterns["attn_pattern"]) if patterns["attn_pattern"] else None

    expert_weights = {}
    router_weights = {}
    attn_weights = {}

    for name, param in model.named_parameters():
        if param.ndim == 3 and fused_re:
            m = fused_re.match(name)
            if m:
                li = int(m.group(1))
                pt = m.group(2)
                n_exp = param.shape[0]
                for ei in range(n_exp):
                    expert_weights.setdefault(li, {}).setdefault(ei, {})[pt] = param.data[ei]
                continue

        if param.ndim != 2:
            continue

        if expert_re:
            m = expert_re.match(name)
            if m:
                li, ei = int(m.group(1)), int(m.group(2))
                pt = m.group(3)
                expert_weights.setdefault(li, {}).setdefault(ei, {})[pt] = param.data
                continue
        if router_re:
            m = router_re.match(name)
            if m:
                router_weights[int(m.group(1))] = param.data
                continue
        if attn_re:
            m = attn_re.match(name)
            if m:
                li = int(m.group(1))
                pt = m.group(2)
                attn_weights.setdefault(li, {})[pt] = param.data
                continue

    n_moe_layers = len(expert_weights)
    print(f"  Found {n_moe_layers} MoE layers, {len(attn_weights)} attn layers")
    if not expert_weights:
        print("  ERROR: No expert weights found!")
        # Print sample keys for debugging
        expert_keys = [k for k in keys if "expert" in k.lower()][:10]
        print(f"  Expert-related keys: {expert_keys}")
        del model; gc.collect()
        return None

    # Sample layers evenly
    all_layers = sorted(expert_weights.keys())
    if max_layers > 0 and len(all_layers) > max_layers:
        indices = np.linspace(0, len(all_layers)-1, max_layers, dtype=int)
        sample_layers = [all_layers[i] for i in indices]
    else:
        sample_layers = all_layers

    # Measure
    print(f"[4] Measuring {len(sample_layers)} layers x {max_experts_per_layer} experts...")
    all_expert_alphas = []
    all_expert_srds = []
    all_expert_psis = []
    all_attn_alphas = []
    all_attn_srds = []
    all_router_srds = []
    all_eprs = []
    layer_results = []

    t0 = time.time()
    for li in sample_layers:
        experts = expert_weights[li]
        expert_ids = sorted(experts.keys())
        if max_experts_per_layer > 0 and len(expert_ids) > max_experts_per_layer:
            idx = np.linspace(0, len(expert_ids)-1, max_experts_per_layer, dtype=int)
            expert_ids = [expert_ids[i] for i in idx]

        layer_alphas = []
        layer_srds = []
        layer_psis = []
        layer_volumes = []
        alignment_weights = []

        for ei in expert_ids:
            expert_vol = 0.0
            for pt, w in experts[ei].items():
                r = measure_expert_weight(w, ei, li, pt, hidden_dim)
                all_expert_alphas.append(r.alpha)
                all_expert_srds.append(r.srd)
                all_expert_psis.append(r.order_parameter)
                layer_alphas.append(r.alpha)
                layer_srds.append(r.srd)
                layer_psis.append(r.order_parameter)
                expert_vol += r.volume
                if len(alignment_weights) < max_experts_per_layer and pt in ("w1", "gate_proj", "gate_up_proj"):
                    alignment_weights.append(w)
            layer_volumes.append(expert_vol)

        epr = compute_epr(layer_volumes) if layer_volumes else 0
        all_eprs.append(epr)

        alignment = 0.0
        if len(alignment_weights) >= 2:
            try:
                alignment = compute_cross_expert_alignment(alignment_weights, top_k=10)
            except Exception:
                pass

        if li in router_weights:
            rw = router_weights[li]
            with torch.no_grad():
                rsv = compute_svd(rw, k=min(64, min(rw.shape)))
                rsr = compute_stable_rank(rsv)
                r_srd = rsr / rw.shape[1] if rw.shape[1] > 0 else 0
                all_router_srds.append(r_srd)

        if li in attn_weights:
            for pt, w in attn_weights[li].items():
                with torch.no_grad():
                    sv = compute_svd(w)
                    a = compute_alpha(sv)
                    sr = compute_stable_rank(sv)
                    srd = sr / hidden_dim
                if not np.isnan(a):
                    all_attn_alphas.append(a)
                all_attn_srds.append(srd)

        alpha_m = np.mean(layer_alphas) if layer_alphas else 0
        alpha_s = np.std(layer_alphas) if layer_alphas else 0
        srd_m = np.mean(layer_srds) if layer_srds else 0
        print(f"  Layer {li:>2}: alpha={alpha_m:.3f}+/-{alpha_s:.3f}, "
              f"SR/d={srd_m:.5f}, EPR={epr:.4f}, align={alignment:.4f}")
        layer_results.append({
            "layer": li,
            "alpha_mean": float(alpha_m),
            "alpha_std": float(alpha_s),
            "srd_mean": float(srd_m),
            "epr": epr,
            "alignment": alignment,
        })

    t1 = time.time()
    print(f"  Measurement time: {t1-t0:.1f}s")

    valid_alphas = [a for a in all_expert_alphas if not np.isnan(a)]
    result = {
        "model": model_id,
        "arch": arch,
        "total_params": total_params,
        "hidden_dim": hidden_dim,
        "intermediate_size": intermediate,
        "num_experts": num_experts,
        "top_k": top_k,
        "num_layers": num_layers,
        "alpha_moe": float(np.mean(valid_alphas)) if valid_alphas else None,
        "alpha_std": float(np.std(valid_alphas)) if valid_alphas else None,
        "alpha_min": float(np.min(valid_alphas)) if valid_alphas else None,
        "alpha_max": float(np.max(valid_alphas)) if valid_alphas else None,
        "srd_moe": float(np.mean(all_expert_srds)) if all_expert_srds else None,
        "srd_std": float(np.std(all_expert_srds)) if all_expert_srds else None,
        "alpha_attn": float(np.mean(all_attn_alphas)) if all_attn_alphas else None,
        "srd_attn": float(np.mean(all_attn_srds)) if all_attn_srds else None,
        "epr_mean": float(np.mean(all_eprs)) if all_eprs else None,
        "psi_moe": float(np.mean(all_expert_psis)) if all_expert_psis else None,
        "router_srd_mean": float(np.mean(all_router_srds)) if all_router_srds else None,
        "layers_measured": len(sample_layers),
        "experts_per_layer_measured": max_experts_per_layer,
        "layer_details": layer_results,
    }

    print(f"\n  SUMMARY:")
    print(f"    Expert alpha: {result['alpha_moe']:.3f} +/- {result['alpha_std']:.3f}")
    if result['alpha_min'] is not None:
        print(f"    Alpha range: [{result['alpha_min']:.2f}, {result['alpha_max']:.2f}]")
    print(f"    Expert SR/d: {result['srd_moe']:.5f}")
    if result['alpha_attn'] is not None:
        print(f"    Attn alpha: {result['alpha_attn']:.3f}, SR/d: {result['srd_attn']:.5f}")
    if result['epr_mean'] is not None:
        print(f"    EPR: {result['epr_mean']:.4f}")
    if result['psi_moe'] is not None:
        print(f"    Psi: {result['psi_moe']:.4f}")
    if result['router_srd_mean'] is not None:
        print(f"    Router SR/d: {result['router_srd_mean']:.5f}")

    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  Saved to {RESULTS_FILE}")

    del model
    gc.collect()
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2: Cross-Model Spectral Comparison")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 70)

    # Model 1: Mixtral-8x7B (large experts)
    r1 = measure_model(
        "mistralai/Mixtral-8x7B-v0.1",
        max_experts_per_layer=8,
        max_layers=8,
    )

    # Model 2: Phi-3.5-MoE (medium experts)
    r2 = measure_model(
        "microsoft/Phi-3.5-MoE-instruct",
        max_experts_per_layer=8,
        max_layers=8,
    )

    # Final comparison table
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON")
    print("=" * 70)
    fmt = "{:<30} {:>8} {:>12} {:>8} {:>8} {:>8}"
    print(fmt.format("Model", "Experts", "Intermediate", "alpha", "SR/d", "Psi"))
    print("-" * 70)
    print(fmt.format("OLMoE-1B-7B (Phase1)", "64", "1024", "1.459", "0.0523", "0.094"))
    if r1:
        print(fmt.format("Mixtral-8x7B", str(r1["num_experts"]), str(r1["intermediate_size"]),
                         f"{r1['alpha_moe']:.3f}", f"{r1['srd_moe']:.4f}", f"{r1['psi_moe']:.3f}"))
    if r2:
        print(fmt.format("Phi-3.5-MoE", str(r2["num_experts"]), str(r2["intermediate_size"]),
                         f"{r2['alpha_moe']:.3f}", f"{r2['srd_moe']:.4f}", f"{r2['psi_moe']:.3f}"))

    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("PHASE 2 COMPLETE")
