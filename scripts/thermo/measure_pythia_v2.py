"""Measure improved thermodynamic state variables from Pythia checkpoints (V2).

Key improvements over V1:
  - Power-law exponent α (Martin & Mahoney): scale-invariant structure metric
  - Stable rank: ||W||²_F / σ₁² (normalized by top SV)
  - Normalized spectral entropy: S / log(min(m,n)) ∈ [0,1]
  - Top-k concentration: fraction of variance in top-k singular values
  - Per-layer type stratification (attention QKV, output, MLP)
  - α_hat = α × log₁₀(λ_max / λ_min): weighted alpha

Usage:
    torchrun --nproc_per_node=8 scripts/thermo/measure_pythia_v2.py \
        --model-size 70m --output results/pythia_v2/pythia_70m.jsonl
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

PYTHIA_CONFIGS = {
    "70m": {
        "hf_repo": "EleutherAI/pythia-70m-deduped",
        "num_params": 70_400_000,
        "hidden_dim": 512,
        "num_layers": 6,
        "peak_lr": 1.0e-3,
        "min_lr": 1.0e-4,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
    },
    "160m": {
        "hf_repo": "EleutherAI/pythia-160m-deduped",
        "num_params": 162_300_000,
        "hidden_dim": 768,
        "num_layers": 12,
        "peak_lr": 6.0e-4,
        "min_lr": 6.0e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
    },
    "410m": {
        "hf_repo": "EleutherAI/pythia-410m-deduped",
        "num_params": 405_300_000,
        "hidden_dim": 1024,
        "num_layers": 24,
        "peak_lr": 3.0e-4,
        "min_lr": 3.0e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
    },
    "1b": {
        "hf_repo": "EleutherAI/pythia-1b-deduped",
        "num_params": 1_011_800_000,
        "hidden_dim": 2048,
        "num_layers": 16,
        "peak_lr": 2.5e-4,
        "min_lr": 2.5e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
    },
    "2.8b": {
        "hf_repo": "EleutherAI/pythia-2.8b-deduped",
        "num_params": 2_775_200_000,
        "hidden_dim": 2560,
        "num_layers": 32,
        "peak_lr": 1.6e-4,
        "min_lr": 1.6e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
    },
    "6.9b": {
        "hf_repo": "EleutherAI/pythia-6.9b-deduped",
        "num_params": 6_857_300_000,
        "hidden_dim": 4096,
        "num_layers": 32,
        "peak_lr": 1.2e-4,
        "min_lr": 1.2e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
    },
}

SAMPLE_STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 10000, 20000, 30000, 40000, 50000, 60000,
    70000, 80000, 90000, 100000, 110000, 120000, 143000,
]


def fit_power_law_alpha(singular_values: np.ndarray) -> dict:
    """Fit power-law exponent α to the singular value spectrum.

    Uses the empirical spectral density (ESD) approach from Martin & Mahoney:
    Rank-ordered singular values should follow σ_i ∝ i^(-1/α) for heavy-tailed spectra.

    We fit in log-log space: log(σ) vs log(rank), slope = -1/α → α = -1/slope.
    Only fit to the bulk (exclude top 5% and bottom 5% to avoid edge effects).
    """
    sv = np.sort(singular_values)[::-1]  # descending
    n = len(sv)
    if n < 10:
        return {"alpha": float("inf"), "alpha_hat": 0.0, "fit_r2": 0.0}

    # Use eigenvalues (λ = σ²) for ESD fitting (standard in RMT literature)
    eigenvalues = sv ** 2
    eig_pos = eigenvalues[eigenvalues > 1e-20]
    n_eig = len(eig_pos)
    if n_eig < 10:
        return {"alpha": float("inf"), "alpha_hat": 0.0, "fit_r2": 0.0}

    # Fit power law to the tail using log-log linear regression
    # Exclude top 2% (potential outliers) and bottom 20% (noise floor)
    start_idx = max(1, int(n_eig * 0.02))
    end_idx = max(start_idx + 5, int(n_eig * 0.80))

    log_rank = np.log10(np.arange(start_idx, end_idx) + 1)
    log_eig = np.log10(eig_pos[start_idx:end_idx])

    if len(log_rank) < 5:
        return {"alpha": float("inf"), "alpha_hat": 0.0, "fit_r2": 0.0}

    # Linear fit: log(λ) = slope * log(rank) + intercept
    # For power-law λ_i ~ i^(-2/α), slope = -2/α → α = -2/slope
    coeffs = np.polyfit(log_rank, log_eig, 1)
    slope = coeffs[0]

    if slope >= -0.01:  # Nearly flat = no power law (random matrix)
        alpha = float("inf")
    else:
        alpha = -2.0 / slope  # Convert from eigenvalue slope to α

    # Clamp to reasonable range
    alpha = min(max(alpha, 1.0), 20.0)

    # R² of the fit
    predicted = np.polyval(coeffs, log_rank)
    ss_res = np.sum((log_eig - predicted) ** 2)
    ss_tot = np.sum((log_eig - np.mean(log_eig)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # α_hat = α × log₁₀(λ_max / λ_min) — Martin & Mahoney's weighted metric
    lambda_max = eig_pos[0]
    lambda_min = eig_pos[-1] if eig_pos[-1] > 1e-20 else eig_pos[max(0, n_eig - 1)]
    log_ratio = np.log10(lambda_max / lambda_min) if lambda_min > 1e-20 else 0.0
    alpha_hat = alpha * log_ratio

    return {
        "alpha": float(alpha),
        "alpha_hat": float(alpha_hat),
        "fit_r2": float(r2),
        "log_lambda_ratio": float(log_ratio),
    }


@torch.no_grad()
def measure_model_v2(model, svd_k: int = 256) -> dict:
    """Compute improved thermodynamic state variables."""
    # Per-layer accumulators
    layer_alphas = []
    layer_alpha_hats = []
    layer_stable_ranks = []
    layer_norm_entropies = []
    layer_concentrations_1 = []
    layer_concentrations_5 = []
    layer_concentrations_10 = []
    layer_n_params = []

    # Global accumulators
    total_params = 0
    vol = 0.0
    weighted_entropy = 0.0
    weighted_norm_entropy = 0.0
    psi_values = []

    # Layer type classification
    attn_alphas = []
    mlp_alphas = []

    for name, param in model.named_parameters():
        n_elem = param.numel()
        total_params += n_elem
        vol += param.data.float().pow(2).sum().item()

        if param.ndim != 2:
            continue

        w = param.data.float()
        m, n = w.shape
        min_dim = min(m, n)
        n_params_layer = m * n

        if min_dim < 4:
            continue

        # Full or truncated SVD
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

        # 1. Power-law α
        pl = fit_power_law_alpha(sv_pos)
        layer_alphas.append(pl["alpha"])
        layer_alpha_hats.append(pl["alpha_hat"])

        # 2. Stable rank: ||W||²_F / σ₁²
        frob_sq = float(w.pow(2).sum().item())
        sigma1_sq = float(sv_pos[0] ** 2)
        stable_rank = frob_sq / sigma1_sq if sigma1_sq > 1e-20 else min_dim
        layer_stable_ranks.append(stable_rank)

        # 3. Spectral entropy (standard)
        p = sv_pos / sv_pos.sum()
        entropy = -np.sum(p * np.log(p + 1e-30))
        weighted_entropy += n_params_layer * entropy

        # 4. Normalized entropy: S / log(min(m,n))
        max_entropy = np.log(min_dim)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        layer_norm_entropies.append(norm_entropy)
        weighted_norm_entropy += n_params_layer * norm_entropy

        # 5. Top-k concentration: Σσ²_top_k / Σσ²_all
        sv_sq = sv_pos ** 2
        total_var = sv_sq.sum()
        if total_var > 0:
            layer_concentrations_1.append(float(sv_sq[0] / total_var))
            layer_concentrations_5.append(float(sv_sq[:5].sum() / total_var) if len(sv_sq) >= 5 else 1.0)
            layer_concentrations_10.append(float(sv_sq[:10].sum() / total_var) if len(sv_sq) >= 10 else 1.0)

        # 6. ψ (keep for backwards compatibility)
        if len(sv_pos) >= 2:
            s1, s2 = float(sv_pos[0]), float(sv_pos[1])
            denom = s1 + s2
            if denom > 1e-10:
                psi_values.append((s1 - s2) / denom)

        layer_n_params.append(n_params_layer)

        # Classify layer type
        if "attention" in name or "query" in name or "key" in name or "value" in name:
            attn_alphas.append(pl["alpha"])
        elif "mlp" in name or "dense_4h" in name or "dense_h_to_4h" in name:
            mlp_alphas.append(pl["alpha"])

    # Global aggregations
    total_2d_params = sum(layer_n_params) if layer_n_params else 1
    weights = np.array(layer_n_params) / total_2d_params if layer_n_params else np.array([1.0])

    return {
        # Original metrics (for comparison)
        "volume": vol,
        "spectral_entropy": weighted_entropy / max(total_2d_params, 1),
        "order_parameter": float(np.mean(psi_values)) if psi_values else 0.0,
        "n_params": total_params,

        # NEW: Power-law metrics (Martin & Mahoney)
        "alpha_mean": float(np.mean(layer_alphas)) if layer_alphas else 0.0,
        "alpha_median": float(np.median(layer_alphas)) if layer_alphas else 0.0,
        "alpha_hat_mean": float(np.mean(layer_alpha_hats)) if layer_alpha_hats else 0.0,
        "alpha_weighted": float(np.average(layer_alphas, weights=weights[:len(layer_alphas)])) if layer_alphas else 0.0,

        # NEW: Stable rank
        "stable_rank_mean": float(np.mean(layer_stable_ranks)) if layer_stable_ranks else 0.0,
        "stable_rank_median": float(np.median(layer_stable_ranks)) if layer_stable_ranks else 0.0,

        # NEW: Normalized entropy (scale-invariant)
        "norm_entropy_mean": float(np.mean(layer_norm_entropies)) if layer_norm_entropies else 0.0,
        "norm_entropy_weighted": weighted_norm_entropy / max(total_2d_params, 1),

        # NEW: Concentration metrics
        "concentration_top1": float(np.mean(layer_concentrations_1)) if layer_concentrations_1 else 0.0,
        "concentration_top5": float(np.mean(layer_concentrations_5)) if layer_concentrations_5 else 0.0,
        "concentration_top10": float(np.mean(layer_concentrations_10)) if layer_concentrations_10 else 0.0,

        # NEW: Per-type alpha (attention vs MLP)
        "alpha_attn": float(np.mean(attn_alphas)) if attn_alphas else 0.0,
        "alpha_mlp": float(np.mean(mlp_alphas)) if mlp_alphas else 0.0,

        # Metadata
        "n_layers_measured": len(layer_alphas),
    }


def cosine_lr_at_step(step: int, config: dict) -> float:
    warmup = config["warmup_steps"]
    total = config["total_steps"]
    max_lr = config["peak_lr"]
    min_lr = config["min_lr"]
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


def cleanup_hf_cache(repo_id: str):
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    repo_dir_name = f"models--{repo_id.replace('/', '--')}"
    repo_cache = Path(hf_home) / "hub" / repo_dir_name
    if repo_cache.exists():
        shutil.rmtree(repo_cache, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Pythia checkpoints (V2 metrics)")
    parser.add_argument("--model-size", required=True, choices=list(PYTHIA_CONFIGS.keys()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--svd-k", type=int, default=256)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PYTHIA_CONFIGS[args.model_size]
    repo_id = config["hf_repo"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    base_hf_home = os.environ.get("HF_HOME", "/fsx/dev/jiaqi/.cache/huggingface")
    per_gpu_hf_home = f"{base_hf_home}_rank{local_rank}"
    os.environ["HF_HOME"] = per_gpu_hf_home
    Path(per_gpu_hf_home).mkdir(parents=True, exist_ok=True)

    steps = [int(s) for s in args.steps.split(",")] if args.steps else SAMPLE_STEPS

    completed_steps = set()
    output_path = Path(args.output)
    if world_size > 1:
        output_path = output_path.with_suffix(f".rank{local_rank}.jsonl")

    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    completed_steps.add(r["step"])

    steps = [s for s in steps if s not in completed_steps]
    my_steps = steps[local_rank::world_size]

    if local_rank == 0:
        print(f"[V2] Measuring Pythia-{args.model_size} ({repo_id})")
        print(f"  Steps: {len(my_steps)} per GPU, {len(steps)} total")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    mode = "a" if args.resume else "w"
    with open(output_path, mode) as f:
        for i, step in enumerate(my_steps):
            revision = f"step{step}"
            lr = cosine_lr_at_step(step, config)
            tokens_b = step * config["batch_size_tokens"] / 1e9

            print(f"[GPU {local_rank}] [{i+1}/{len(my_steps)}] step={step}")
            t0 = time.time()

            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id, revision=revision,
                    torch_dtype=torch.float32,
                    device_map=device, trust_remote_code=True,
                )

                measurements = measure_model_v2(model, svd_k=args.svd_k)

                record = {
                    "step": step,
                    "tokens_b": round(tokens_b, 2),
                    "model_size": args.model_size,
                    "lr": lr,
                    "hidden_dim": config["hidden_dim"],
                    **measurements,
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

                elapsed = time.time() - t0
                print(f"  α={measurements['alpha_mean']:.2f} "
                      f"α̂={measurements['alpha_hat_mean']:.1f} "
                      f"SR={measurements['stable_rank_mean']:.1f} "
                      f"S_n={measurements['norm_entropy_weighted']:.4f} "
                      f"C₁₀={measurements['concentration_top10']:.3f} "
                      f"({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                f.write(json.dumps({"step": step, "error": str(e)}) + "\n")
                f.flush()
            finally:
                if "model" in locals():
                    del model
                torch.cuda.empty_cache()
                cleanup_hf_cache(repo_id)

    print(f"[GPU {local_rank}] Done: {output_path}")

    # Barrier: wait for all ranks to finish before merging
    if world_size > 1:
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend="gloo")
            dist.barrier()
        except Exception:
            import time as _t
            _t.sleep(30)

        if local_rank == 0:
            merged_path = Path(args.output)
            all_records = []
            for rank in range(world_size):
                shard = merged_path.with_suffix(f".rank{rank}.jsonl")
                if shard.exists():
                    with open(shard) as sf:
                        for line in sf:
                            all_records.append(json.loads(line))
            all_records.sort(key=lambda r: r.get("step", 0))
            with open(merged_path, "w") as mf:
                for r in all_records:
                    mf.write(json.dumps(r) + "\n")
            print(f"Merged {len(all_records)} records → {merged_path}")


if __name__ == "__main__":
    main()
