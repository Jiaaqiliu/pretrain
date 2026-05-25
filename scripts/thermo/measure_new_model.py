"""Measure spectral properties of arbitrary HuggingFace models.

Computes SR/d, α, α_attn, α_mlp for any dense transformer model.
Used for hold-out validation of the SR/d universal formula.

Usage:
    python scripts/thermo/measure_new_model.py \
        --model mistralai/Mistral-7B-v0.3 \
        --output results/mistral_v2/mistral_7b.jsonl

    python scripts/thermo/measure_new_model.py \
        --model google/gemma-2-9b \
        --output results/gemma_v2/gemma_9b.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def get_model_info(model):
    """Extract architecture info from a HuggingFace model."""
    config = model.config
    hidden_dim = getattr(config, 'hidden_size', None)
    num_layers = getattr(config, 'num_hidden_layers', None)
    num_params = sum(p.numel() for p in model.parameters())
    model_type = getattr(config, 'model_type', 'unknown')
    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_params": num_params,
        "model_type": model_type,
    }


def classify_layer(name):
    """Classify a layer as attention, mlp, or other."""
    name_lower = name.lower()
    attn_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv', 'attn', 'self_attn',
                 'query', 'key', 'value', 'attention']
    mlp_keys = ['mlp', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2',
                'dense_h_to_4h', 'dense_4h_to_h', 'feed_forward']

    for k in attn_keys:
        if k in name_lower:
            return "attn"
    for k in mlp_keys:
        if k in name_lower:
            return "mlp"
    return "other"


def compute_layer_metrics(w, k=256):
    """Compute spectral metrics for a single weight matrix."""
    m, n = w.shape
    min_dim = min(m, n)

    with torch.no_grad():
        frob_sq = (w * w).sum().item()

        if min_dim <= 1024:
            sv = torch.linalg.svdvals(w).cpu().numpy()
        else:
            rank = min(k, min_dim - 1)
            omega = torch.randn(n, rank + 16, device=w.device, dtype=torch.float32)
            y = w @ omega
            q, _ = torch.linalg.qr(y)
            b = q.T @ w
            sv = torch.linalg.svdvals(b)[:rank].cpu().numpy()

    sigma1_sq = float(sv[0]) ** 2
    stable_rank = frob_sq / (sigma1_sq + 1e-12)

    sv_pos = sv[sv > 1e-10]
    if len(sv_pos) < 10:
        return {"stable_rank": stable_rank, "alpha": None}

    eigenvalues = sv_pos ** 2
    eig_pos = eigenvalues[eigenvalues > 1e-20]
    n_eig = len(eig_pos)

    start_idx = max(1, int(n_eig * 0.02))
    end_idx = max(start_idx + 5, int(n_eig * 0.80))

    log_rank = np.log10(np.arange(start_idx, end_idx) + 1)
    log_eig = np.log10(eig_pos[start_idx:end_idx])

    if len(log_rank) < 5:
        return {"stable_rank": stable_rank, "alpha": None}

    coeffs = np.polyfit(log_rank, log_eig, 1)
    slope = coeffs[0]
    alpha = -2.0 / slope if slope < -0.01 else 20.0
    alpha = min(max(alpha, 1.0), 20.0)

    p = sv_pos ** 2 / np.sum(sv_pos ** 2)
    concentration_top10 = float(np.sum(p[:10]))

    return {
        "stable_rank": stable_rank,
        "alpha": alpha,
        "concentration_top10": concentration_top10,
    }


def measure_model(model, hidden_dim=None):
    """Measure all spectral metrics for a model."""
    info = get_model_info(model)
    if hidden_dim is None:
        hidden_dim = info["hidden_dim"]

    all_metrics = []
    attn_alphas = []
    mlp_alphas = []
    stable_ranks = []

    named_params = list(model.named_parameters())
    layers_2d = [(name, param) for name, param in named_params
                 if param.ndim == 2 and min(param.shape) >= 64]

    print(f"  Measuring {len(layers_2d)} 2D layers...")

    for i, (name, param) in enumerate(layers_2d):
        if (i + 1) % 20 == 0:
            print(f"    Layer {i+1}/{len(layers_2d)}: {name} ({param.shape[0]}x{param.shape[1]})")

        w = param.data.float()
        if w.device.type == 'meta':
            continue
        if w.device.type == 'cpu' and torch.cuda.is_available():
            w = w.cuda()

        metrics = compute_layer_metrics(w)
        layer_type = classify_layer(name)

        metrics["name"] = name
        metrics["layer_type"] = layer_type
        metrics["shape"] = list(param.shape)
        metrics["num_params"] = param.numel()
        all_metrics.append(metrics)

        if metrics["stable_rank"] is not None:
            stable_ranks.append(metrics["stable_rank"])

        if metrics["alpha"] is not None:
            if layer_type == "attn":
                attn_alphas.append(metrics["alpha"])
            elif layer_type == "mlp":
                mlp_alphas.append(metrics["alpha"])

    all_alphas = [m["alpha"] for m in all_metrics if m["alpha"] is not None]

    result = {
        "model_name": info.get("model_type", "unknown"),
        "hidden_dim": hidden_dim,
        "num_params": info["num_params"],
        "num_layers": info["num_layers"],
        "n_layers_measured": len(layers_2d),
        "stable_rank_mean": float(np.mean(stable_ranks)) if stable_ranks else None,
        "sr_over_d": float(np.mean(stable_ranks)) / hidden_dim if stable_ranks else None,
        "alpha_mean": float(np.mean(all_alphas)) if all_alphas else None,
        "alpha_median": float(np.median(all_alphas)) if all_alphas else None,
        "alpha_attn": float(np.mean(attn_alphas)) if attn_alphas else None,
        "alpha_mlp": float(np.mean(mlp_alphas)) if mlp_alphas else None,
        "n_attn_layers": len(attn_alphas),
        "n_mlp_layers": len(mlp_alphas),
    }

    if attn_alphas and mlp_alphas:
        result["mlp_attn_gap"] = result["alpha_mlp"] - result["alpha_attn"]

    predicted_sr_d = 0.040 + 0.61 / np.sqrt(hidden_dim)
    result["predicted_sr_d"] = float(predicted_sr_d)
    if result["sr_over_d"] is not None:
        result["prediction_error"] = result["sr_over_d"] - predicted_sr_d

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dim")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    print(f"=== Measuring: {args.model} ===")
    t_start = time.time()

    from transformers import AutoModelForCausalLM

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    load_dtype = dtype_map[args.dtype]

    print(f"  Loading model in {args.dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=load_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    load_time = time.time() - t_start
    print(f"  Loaded in {load_time:.1f}s")

    result = measure_model(model, hidden_dim=args.hidden_dim)
    result["model_id"] = args.model
    result["load_dtype"] = args.dtype
    result["elapsed_s"] = time.time() - t_start

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json.dumps(result, indent=2) + "\n")

    print(f"\n=== Results for {args.model} ===")
    print(f"  Hidden dim: {result['hidden_dim']}")
    print(f"  Params: {result['num_params']/1e9:.2f}B")
    print(f"  SR/d (measured): {result['sr_over_d']:.4f}")
    print(f"  SR/d (predicted): {result['predicted_sr_d']:.4f}")
    print(f"  Prediction error: {result.get('prediction_error', 'N/A')}")
    print(f"  α (mean): {result['alpha_mean']:.2f}")
    print(f"  α_attn: {result['alpha_attn']:.2f}" if result['alpha_attn'] else "  α_attn: N/A")
    print(f"  α_mlp: {result['alpha_mlp']:.2f}" if result['alpha_mlp'] else "  α_mlp: N/A")
    if result.get('mlp_attn_gap'):
        print(f"  MLP/Attn gap: {result['mlp_attn_gap']:.2f}")
    print(f"  Total time: {result['elapsed_s']:.1f}s")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
