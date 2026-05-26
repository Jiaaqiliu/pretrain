"""Measure per-layer spectral properties for heatmap visualization.

Outputs per-layer α and SR for every checkpoint, enabling:
  - α heatmap (x=layer, y=step, color=α)
  - SR heatmap (x=layer, y=step, color=SR/d)
  - Layer-type coloring (attn vs MLP)

Usage:
    python scripts/thermo/measure_perlayer_heatmap.py \
        --model-size 1b \
        --output results/heatmap/pythia_1b_perlayer.jsonl

    python scripts/thermo/measure_perlayer_heatmap.py \
        --model-size 6.9b \
        --output results/heatmap/pythia_6.9b_perlayer.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

PYTHIA_CONFIGS = {
    "70m": {
        "hf_repo": "EleutherAI/pythia-70m-deduped",
        "hidden_dim": 512,
        "total_steps": 143_000,
        "checkpoints": list(range(0, 144_000, 6_000)),
    },
    "1b": {
        "hf_repo": "EleutherAI/pythia-1b-deduped",
        "hidden_dim": 2048,
        "total_steps": 143_000,
        "checkpoints": list(range(0, 144_000, 6_000)),
    },
    "2.8b": {
        "hf_repo": "EleutherAI/pythia-2.8b-deduped",
        "hidden_dim": 2560,
        "total_steps": 143_000,
        "checkpoints": list(range(0, 144_000, 6_000)),
    },
    "6.9b": {
        "hf_repo": "EleutherAI/pythia-6.9b-deduped",
        "hidden_dim": 4096,
        "total_steps": 143_000,
        "checkpoints": list(range(0, 144_000, 6_000)),
    },
}

OLMO2_CONFIGS = {
    "13b": {
        "hf_repo": "allenai/OLMo-2-1124-13B",
        "hidden_dim": 5120,
        "checkpoints": None,
    },
}


def classify_layer(name):
    name_lower = name.lower()
    attn_keys = ['attention', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    mlp_keys = ['mlp', 'dense_4h', 'dense_h_to_4h', 'gate_proj', 'up_proj', 'down_proj']
    for k in attn_keys:
        if k in name_lower:
            return 'attn'
    for k in mlp_keys:
        if k in name_lower:
            return 'mlp'
    return 'other'


def fit_alpha(sv_pos):
    eigenvalues = sv_pos ** 2
    eig_pos = eigenvalues[eigenvalues > 1e-20]
    n_eig = len(eig_pos)
    if n_eig < 10:
        return 20.0

    start_idx = max(1, int(n_eig * 0.02))
    end_idx = max(start_idx + 5, int(n_eig * 0.80))

    log_rank = np.log10(np.arange(start_idx, end_idx) + 1)
    log_eig = np.log10(eig_pos[start_idx:end_idx])

    if len(log_rank) < 5:
        return 20.0

    coeffs = np.polyfit(log_rank, log_eig, 1)
    slope = coeffs[0]
    alpha = -2.0 / slope if slope < -0.01 else 20.0
    return min(max(alpha, 1.0), 20.0)


@torch.no_grad()
def measure_perlayer(model, hidden_dim, svd_k=256):
    """Measure α and SR for every 2D layer."""
    layers = []

    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        m, n = param.shape
        min_dim = min(m, n)
        if min_dim < 4:
            continue

        w = param.data.float()

        # SVD
        if min_dim <= 2048:
            sv = torch.linalg.svdvals(w).cpu().numpy()
        else:
            actual_k = min(svd_k, min_dim)
            omega = torch.randn(n, actual_k + 16, device=w.device, dtype=torch.float32)
            y = w @ omega
            q, _ = torch.linalg.qr(y)
            z = w.T @ q
            q, _ = torch.linalg.qr(w @ z)
            b = q.T @ w
            sv = torch.linalg.svdvals(b)[:actual_k].cpu().numpy()

        sv_pos = sv[sv > 1e-10]
        if len(sv_pos) < 4:
            continue

        # Alpha
        alpha = fit_alpha(sv_pos)

        # Stable rank
        frob_sq = float(w.pow(2).sum().item())
        sigma1_sq = float(sv_pos[0] ** 2)
        sr = frob_sq / sigma1_sq if sigma1_sq > 1e-20 else min_dim
        sr_d = sr / hidden_dim

        # Concentration
        sv_sq = sv_pos ** 2
        total_var = sv_sq.sum()
        c10 = float(sv_sq[:10].sum() / total_var) if total_var > 0 and len(sv_sq) >= 10 else 1.0

        layer_type = classify_layer(name)

        layers.append({
            "name": name,
            "shape": [m, n],
            "type": layer_type,
            "alpha": round(float(alpha), 3),
            "sr": round(float(sr), 2),
            "sr_d": round(float(sr_d), 5),
            "c10": round(float(c10), 4),
        })

    return layers


def measure_pythia(config, output_path, max_ckpts=None):
    from transformers import AutoModelForCausalLM
    import shutil

    hf_repo = config["hf_repo"]
    hidden_dim = config["hidden_dim"]
    checkpoints = config["checkpoints"]
    if max_ckpts:
        checkpoints = checkpoints[:max_ckpts]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    cache_dir = os.environ.get("HF_HOME", None)

    for i, step in enumerate(checkpoints):
        t0 = time.time()
        revision = f"step{step}"
        print(f"  [{i+1}/{len(checkpoints)}] Loading {hf_repo} @ {revision}...")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_repo, revision=revision, cache_dir=cache_dir
            )
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
        except Exception as e:
            print(f"    ERROR loading: {e}")
            continue

        layers = measure_perlayer(model, hidden_dim)

        record = {
            "step": step,
            "model": hf_repo,
            "hidden_dim": hidden_dim,
            "n_layers": len(layers),
            "layers": layers,
            "alpha_mean": round(float(np.mean([l["alpha"] for l in layers])), 3),
            "sr_d_mean": round(float(np.mean([l["sr_d"] for l in layers])), 5),
        }
        results.append(record)

        elapsed = time.time() - t0
        print(f"    Done: {len(layers)} layers, α_mean={record['alpha_mean']:.2f}, "
              f"SR/d={record['sr_d_mean']:.4f}, time={elapsed:.1f}s")

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

        # Clean HF cache to avoid filling disk
        if cache_dir:
            snapshot_dir = Path(cache_dir) / "hub" / f"models--{hf_repo.replace('/', '--')}" / "snapshots"
            if snapshot_dir.exists():
                for d in snapshot_dir.iterdir():
                    shutil.rmtree(d, ignore_errors=True)

    # Save results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} records to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=list(PYTHIA_CONFIGS.keys()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-ckpts", type=int, default=None)
    args = parser.parse_args()

    config = PYTHIA_CONFIGS[args.model_size]
    print(f"=== Per-Layer Heatmap Measurement: {config['hf_repo']} ===")
    print(f"  Checkpoints: {len(config['checkpoints'])} (max={args.max_ckpts})")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print()

    measure_pythia(config, args.output, args.max_ckpts)


if __name__ == "__main__":
    main()
