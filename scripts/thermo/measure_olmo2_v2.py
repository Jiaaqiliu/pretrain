"""Measure V2 thermodynamic metrics from OLMo-2 checkpoints.

Reuses the V2 measurement functions for OLMo-2 models on HuggingFace.

Usage:
    torchrun --nproc_per_node=8 scripts/thermo/measure_olmo2_v2.py \
        --model-size 1B --output results/olmo2_v2/olmo2_1b.jsonl
"""

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch

OLMO2_CONFIGS = {
    "1B": {
        "hf_repo": "allenai/OLMo-2-0425-1B",
        "num_params": 1_000_000_000,
        "hidden_dim": 2048,
        "num_layers": 16,
        "weight_decay": 0.1,
        "peak_lr": 4e-4,
    },
    "7B": {
        "hf_repo": "allenai/OLMo-2-1124-7B",
        "num_params": 7_000_000_000,
        "hidden_dim": 4096,
        "num_layers": 32,
        "weight_decay": 0.1,
        "peak_lr": 3e-4,
    },
    "13B": {
        "hf_repo": "allenai/OLMo-2-1124-13B",
        "num_params": 13_000_000_000,
        "hidden_dim": 5120,
        "num_layers": 40,
        "weight_decay": 0.1,
        "peak_lr": 3e-4,
    },
}


def discover_revisions(repo_id: str, max_count: int = 25) -> list[dict]:
    """Discover checkpoint revisions, sample evenly."""
    from huggingface_hub import list_repo_refs

    refs = list_repo_refs(repo_id)
    checkpoints = []
    step_pattern = re.compile(r"step(\d+)")

    for branch in refs.branches:
        match = step_pattern.search(branch.name)
        if not match:
            continue
        step = int(match.group(1))
        checkpoints.append({"revision": branch.name, "step": step})

    checkpoints.sort(key=lambda x: x["step"])

    if len(checkpoints) > max_count:
        indices = np.linspace(0, len(checkpoints) - 1, max_count, dtype=int)
        checkpoints = [checkpoints[i] for i in indices]

    return checkpoints


def fit_power_law_alpha(singular_values: np.ndarray) -> dict:
    """Fit power-law exponent α."""
    sv = np.sort(singular_values)[::-1]
    n = len(sv)
    if n < 10:
        return {"alpha": float("inf"), "alpha_hat": 0.0}

    eigenvalues = sv ** 2
    eig_pos = eigenvalues[eigenvalues > 1e-20]
    n_eig = len(eig_pos)
    if n_eig < 10:
        return {"alpha": float("inf"), "alpha_hat": 0.0}

    start_idx = max(1, int(n_eig * 0.02))
    end_idx = max(start_idx + 5, int(n_eig * 0.80))

    log_rank = np.log10(np.arange(start_idx, end_idx) + 1)
    log_eig = np.log10(eig_pos[start_idx:end_idx])

    if len(log_rank) < 5:
        return {"alpha": float("inf"), "alpha_hat": 0.0}

    coeffs = np.polyfit(log_rank, log_eig, 1)
    slope = coeffs[0]

    alpha = -2.0 / slope if slope < -0.01 else 20.0
    alpha = min(max(alpha, 1.0), 20.0)

    lambda_max = eig_pos[0]
    lambda_min = eig_pos[-1] if eig_pos[-1] > 1e-20 else eig_pos[max(0, n_eig - 1)]
    log_ratio = np.log10(lambda_max / lambda_min) if lambda_min > 1e-20 else 0.0
    alpha_hat = alpha * log_ratio

    return {"alpha": float(alpha), "alpha_hat": float(alpha_hat)}


@torch.no_grad()
def measure_model_v2(model, svd_k: int = 256) -> dict:
    """Compute V2 metrics."""
    layer_alphas = []
    layer_stable_ranks = []
    layer_concentrations_10 = []
    layer_n_params = []
    attn_alphas = []
    mlp_alphas = []
    total_params = 0
    vol = 0.0

    for name, param in model.named_parameters():
        total_params += param.numel()
        vol += param.data.float().pow(2).sum().item()

        if param.ndim != 2:
            continue

        w = param.data.float()
        m, n = w.shape
        min_dim = min(m, n)

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

        frob_sq = float(w.pow(2).sum().item())
        sigma1_sq = float(sv_pos[0] ** 2)
        stable_rank = frob_sq / sigma1_sq if sigma1_sq > 1e-20 else min_dim
        layer_stable_ranks.append(stable_rank)

        sv_sq = sv_pos ** 2
        total_var = sv_sq.sum()
        if total_var > 0 and len(sv_sq) >= 10:
            layer_concentrations_10.append(float(sv_sq[:10].sum() / total_var))

        layer_n_params.append(m * n)

        if "self_attn" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
            attn_alphas.append(pl["alpha"])
        elif "mlp" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            mlp_alphas.append(pl["alpha"])

    return {
        "volume": vol,
        "n_params": total_params,
        "alpha_mean": float(np.mean(layer_alphas)) if layer_alphas else 0.0,
        "alpha_median": float(np.median(layer_alphas)) if layer_alphas else 0.0,
        "stable_rank_mean": float(np.mean(layer_stable_ranks)) if layer_stable_ranks else 0.0,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=list(OLMO2_CONFIGS.keys()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--svd-k", type=int, default=256)
    parser.add_argument("--max-checkpoints", type=int, default=25)
    args = parser.parse_args()

    config = OLMO2_CONFIGS[args.model_size]
    repo_id = config["hf_repo"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    base_hf_home = os.environ.get("HF_HOME", "/fsx/dev/jiaqi/.cache/huggingface")
    per_gpu_hf_home = f"{base_hf_home}_rank{local_rank}"
    os.environ["HF_HOME"] = per_gpu_hf_home
    Path(per_gpu_hf_home).mkdir(parents=True, exist_ok=True)

    if local_rank == 0:
        print(f"[V2] Measuring OLMo-2-{args.model_size} from {repo_id}")
        print(f"Discovering revisions...")

    checkpoints = discover_revisions(repo_id, max_count=args.max_checkpoints)
    if local_rank == 0:
        print(f"Found {len(checkpoints)} checkpoints to measure")

    my_ckpts = checkpoints[local_rank::world_size]
    output_path = Path(args.output)
    if world_size > 1:
        output_path = output_path.with_suffix(f".rank{local_rank}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    print(f"[GPU {local_rank}] Processing {len(my_ckpts)} checkpoints")

    with open(output_path, "w") as f:
        for i, ckpt in enumerate(my_ckpts):
            step = ckpt["step"]
            revision = ckpt["revision"]
            print(f"[GPU {local_rank}] [{i+1}/{len(my_ckpts)}] step={step}")
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
                    "revision": revision,
                    "model_name": f"OLMo-2-{args.model_size}",
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
                f.write(json.dumps({"step": step, "error": str(e)}) + "\n")
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
            time.sleep(30)

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
