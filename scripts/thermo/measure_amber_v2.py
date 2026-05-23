"""Measure V2 thermodynamic metrics from LLM360/Amber checkpoints.

LLM360/Amber: 7B model, 360 checkpoints (ckpt_000 to ckpt_359), 1.26T tokens.
This gives us the DENSEST training dynamics data for a 7B model.

Usage:
    torchrun --nproc_per_node=8 scripts/thermo/measure_amber_v2.py \
        --output results/amber_v2/amber_7b.jsonl \
        --sample 25
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

AMBER_CONFIG = {
    "hf_repo": "LLM360/Amber",
    "num_params": 6_738_000_000,
    "hidden_dim": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "total_checkpoints": 360,
    "total_tokens": 1_260_000_000_000,  # 1.26T
    "weight_decay": 0.1,
    "peak_lr": 2.0e-4,
    "batch_size_tokens": 4_194_304,  # 4M
}

# Sample 25 checkpoints spanning full training
SAMPLE_CKPTS = [0, 1, 5, 10, 20, 40, 60, 80, 100, 120, 140,
                160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 350, 355, 358, 359]

# Dense option: every 10th checkpoint (36 total)
DENSE_CKPTS = list(range(0, 360, 10)) + [359]


def fit_power_law_alpha(singular_values: np.ndarray) -> dict:
    """Fit power-law exponent α to the singular value spectrum."""
    sv = np.sort(singular_values)[::-1]
    n = len(sv)
    if n < 10:
        return {"alpha": float("inf"), "alpha_hat": 0.0, "fit_r2": 0.0}

    eigenvalues = sv ** 2
    eig_pos = eigenvalues[eigenvalues > 1e-20]
    n_eig = len(eig_pos)
    if n_eig < 10:
        return {"alpha": float("inf"), "alpha_hat": 0.0, "fit_r2": 0.0}

    start_idx = max(1, int(n_eig * 0.02))
    end_idx = max(start_idx + 5, int(n_eig * 0.80))

    log_rank = np.log10(np.arange(start_idx, end_idx) + 1)
    log_eig = np.log10(eig_pos[start_idx:end_idx])

    if len(log_rank) < 5:
        return {"alpha": float("inf"), "alpha_hat": 0.0, "fit_r2": 0.0}

    coeffs = np.polyfit(log_rank, log_eig, 1)
    slope = coeffs[0]

    if slope >= -0.01:
        alpha = float("inf")
    else:
        alpha = -2.0 / slope

    alpha = min(max(alpha, 1.0), 20.0)

    predicted = np.polyval(coeffs, log_rank)
    ss_res = np.sum((log_eig - predicted) ** 2)
    ss_tot = np.sum((log_eig - np.mean(log_eig)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    lambda_max = eig_pos[0]
    lambda_min = eig_pos[-1] if eig_pos[-1] > 1e-20 else eig_pos[max(0, n_eig - 1)]
    log_ratio = np.log10(lambda_max / lambda_min) if lambda_min > 1e-20 else 0.0
    alpha_hat = alpha * log_ratio

    return {"alpha": float(alpha), "alpha_hat": float(alpha_hat), "fit_r2": float(r2)}


@torch.no_grad()
def measure_model_v2(model, svd_k: int = 256) -> dict:
    """Compute V2 thermodynamic metrics."""
    layer_alphas = []
    layer_alpha_hats = []
    layer_stable_ranks = []
    layer_norm_entropies = []
    layer_concentrations_10 = []
    layer_n_params = []
    attn_alphas = []
    mlp_alphas = []

    total_params = 0
    vol = 0.0
    weighted_entropy = 0.0

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

        pl = fit_power_law_alpha(sv_pos)
        layer_alphas.append(pl["alpha"])
        layer_alpha_hats.append(pl["alpha_hat"])

        frob_sq = float(w.pow(2).sum().item())
        sigma1_sq = float(sv_pos[0] ** 2)
        stable_rank = frob_sq / sigma1_sq if sigma1_sq > 1e-20 else min_dim
        layer_stable_ranks.append(stable_rank)

        p = sv_pos / sv_pos.sum()
        entropy = -np.sum(p * np.log(p + 1e-30))
        weighted_entropy += n_params_layer * entropy
        max_entropy = np.log(min_dim)
        layer_norm_entropies.append(entropy / max_entropy if max_entropy > 0 else 0.0)

        sv_sq = sv_pos ** 2
        total_var = sv_sq.sum()
        if total_var > 0 and len(sv_sq) >= 10:
            layer_concentrations_10.append(float(sv_sq[:10].sum() / total_var))

        layer_n_params.append(n_params_layer)

        if "self_attn" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
            attn_alphas.append(pl["alpha"])
        elif "mlp" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            mlp_alphas.append(pl["alpha"])

    total_2d_params = sum(layer_n_params) if layer_n_params else 1

    return {
        "volume": vol,
        "spectral_entropy": weighted_entropy / max(total_2d_params, 1),
        "n_params": total_params,
        "alpha_mean": float(np.mean(layer_alphas)) if layer_alphas else 0.0,
        "alpha_median": float(np.median(layer_alphas)) if layer_alphas else 0.0,
        "alpha_hat_mean": float(np.mean(layer_alpha_hats)) if layer_alpha_hats else 0.0,
        "stable_rank_mean": float(np.mean(layer_stable_ranks)) if layer_stable_ranks else 0.0,
        "stable_rank_median": float(np.median(layer_stable_ranks)) if layer_stable_ranks else 0.0,
        "norm_entropy_mean": float(np.mean(layer_norm_entropies)) if layer_norm_entropies else 0.0,
        "concentration_top10": float(np.mean(layer_concentrations_10)) if layer_concentrations_10 else 0.0,
        "alpha_attn": float(np.mean(attn_alphas)) if attn_alphas else 0.0,
        "alpha_mlp": float(np.mean(mlp_alphas)) if mlp_alphas else 0.0,
        "n_layers_measured": len(layer_alphas),
    }


def cleanup_hf_cache(repo_id: str):
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    repo_dir_name = f"models--{repo_id.replace('/', '--')}"
    repo_cache = Path(hf_home) / "hub" / repo_dir_name
    if repo_cache.exists():
        shutil.rmtree(repo_cache, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--svd-k", type=int, default=256)
    parser.add_argument("--sample", type=int, default=25,
                        help="Number of checkpoints to sample (25 or 36 for dense)")
    parser.add_argument("--ckpts", type=str, default=None,
                        help="Comma-separated checkpoint indices")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = AMBER_CONFIG
    repo_id = config["hf_repo"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    base_hf_home = os.environ.get("HF_HOME", "/fsx/dev/jiaqi/.cache/huggingface")
    per_gpu_hf_home = f"{base_hf_home}_rank{local_rank}"
    os.environ["HF_HOME"] = per_gpu_hf_home
    Path(per_gpu_hf_home).mkdir(parents=True, exist_ok=True)

    if args.ckpts:
        ckpts = [int(c) for c in args.ckpts.split(",")]
    elif args.sample >= 36:
        ckpts = DENSE_CKPTS
    else:
        ckpts = SAMPLE_CKPTS

    completed = set()
    output_path = Path(args.output)
    if world_size > 1:
        output_path = output_path.with_suffix(f".rank{local_rank}.jsonl")

    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    completed.add(r.get("ckpt_idx", -1))

    ckpts = [c for c in ckpts if c not in completed]
    my_ckpts = ckpts[local_rank::world_size]

    if local_rank == 0:
        print(f"[V2] Measuring LLM360/Amber ({len(ckpts)} checkpoints)")
        print(f"  World size: {world_size}, per GPU: {len(my_ckpts)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    tokens_per_ckpt = config["total_tokens"] / config["total_checkpoints"]

    mode = "a" if args.resume else "w"
    with open(output_path, mode) as f:
        for i, ckpt_idx in enumerate(my_ckpts):
            revision = f"ckpt_{ckpt_idx:03d}"
            tokens_b = (ckpt_idx + 1) * tokens_per_ckpt / 1e9

            print(f"[GPU {local_rank}] [{i+1}/{len(my_ckpts)}] {revision} ({tokens_b:.0f}B tokens)")
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
                    "ckpt_idx": ckpt_idx,
                    "revision": revision,
                    "tokens_b": round(tokens_b, 1),
                    "tokens_per_param": round(tokens_b * 1e9 / config["num_params"], 1),
                    "model_name": "LLM360/Amber-7B",
                    "hidden_dim": config["hidden_dim"],
                    **measurements,
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

                elapsed = time.time() - t0
                print(f"  α={measurements['alpha_mean']:.2f} "
                      f"SR={measurements['stable_rank_mean']:.0f} "
                      f"C₁₀={measurements['concentration_top10']:.3f} "
                      f"({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                f.write(json.dumps({"ckpt_idx": ckpt_idx, "revision": revision, "error": str(e)}) + "\n")
                f.flush()
            finally:
                if "model" in locals():
                    del model
                torch.cuda.empty_cache()
                cleanup_hf_cache(repo_id)

    print(f"[GPU {local_rank}] Done: {output_path}")

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
            all_records.sort(key=lambda r: r.get("ckpt_idx", 0))
            with open(merged_path, "w") as mf:
                for r in all_records:
                    mf.write(json.dumps(r) + "\n")
            print(f"Merged {len(all_records)} records → {merged_path}")


if __name__ == "__main__":
    main()
