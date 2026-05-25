"""Measure V2 spectral metrics from LLM360/K2-V2 (70B) checkpoints.

K2-V2 is a 70B dense LLaMA-architecture model with 54 intermediate checkpoints.

Usage:
    torchrun --nproc_per_node=8 scripts/thermo/measure_k2_v2.py \
        --output results/k2_v2/k2_70b.jsonl --max-checkpoints 25
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

K2_CONFIG = {
    "hf_repo": "LLM360/K2",
    "num_params": 65_000_000_000,
    "hidden_dim": 8192,
    "num_layers": 80,
    "num_heads": 64,
    "intermediate_size": 22016,
    "vocab_size": 32032,
}


def discover_revisions(repo_id: str, max_count: int = 25) -> list[dict]:
    """Discover checkpoint branches for K2 (ckpt_XXX format)."""
    from huggingface_hub import list_repo_refs

    refs = list_repo_refs(repo_id)
    checkpoints = []
    step_pattern = re.compile(r"ckpt_(\d+)")

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
def measure_model_v2(model, svd_k: int = 256, hidden_dim: int = 8192) -> dict:
    """Compute V2 metrics layer-by-layer to minimize memory."""
    layer_alphas = []
    layer_stable_ranks = []
    layer_concentrations_10 = []
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
        "sr_over_d": float(np.mean(layer_stable_ranks)) / hidden_dim if layer_stable_ranks else 0.0,
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--svd-k", type=int, default=256)
    parser.add_argument("--max-checkpoints", type=int, default=25)
    args = parser.parse_args()

    config = K2_CONFIG
    repo_id = config["hf_repo"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    base_hf_home = os.environ.get("HF_HOME", "/fsx/dev/jiaqi/.cache/huggingface")
    per_gpu_hf_home = f"{base_hf_home}_rank{local_rank}"
    os.environ["HF_HOME"] = per_gpu_hf_home
    Path(per_gpu_hf_home).mkdir(parents=True, exist_ok=True)

    if local_rank == 0:
        print(f"[V2] Measuring K2-V2 70B from {repo_id}")
        print(f"Discovering revisions...")

    checkpoints = discover_revisions(repo_id, max_count=args.max_checkpoints)
    if local_rank == 0:
        print(f"Found {len(checkpoints)} checkpoints to measure")
        for c in checkpoints[:5]:
            print(f"  {c}")

    my_ckpts = checkpoints[local_rank::world_size]
    output_path = Path(args.output)
    if world_size > 1:
        output_path = output_path.with_suffix(f".rank{local_rank}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For 70B: use device_map="auto" to shard across available GPUs
    # Each rank gets its own checkpoints but uses only its own GPU
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    print(f"[GPU {local_rank}] Processing {len(my_ckpts)} checkpoints")

    with open(output_path, "w") as f:
        for i, ckpt in enumerate(my_ckpts):
            step = ckpt["step"]
            revision = ckpt["revision"]
            print(f"[GPU {local_rank}] [{i+1}/{len(my_ckpts)}] step={step} rev={revision}")
            t0 = time.time()

            try:
                from transformers import AutoModelForCausalLM

                # 70B in fp32 = 280GB, too large for single GPU
                # Load in fp16 (140GB = 2 GPUs) then convert layers to fp32 during measurement
                # Actually: load with device_map to single GPU in fp16, measure layer-by-layer in fp32
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id, revision=revision,
                    torch_dtype=torch.float16,
                    device_map=device,
                    trust_remote_code=True,
                )

                measurements = measure_model_v2(model, svd_k=args.svd_k,
                                                hidden_dim=config["hidden_dim"])
                record = {
                    "step": step,
                    "revision": revision,
                    "model_name": "K2-65B",
                    "hidden_dim": config["hidden_dim"],
                    **measurements,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                elapsed = time.time() - t0
                print(f"  α={measurements['alpha_mean']:.2f} "
                      f"SR={measurements['stable_rank_mean']:.0f} "
                      f"SR/d={measurements['sr_over_d']:.4f} "
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
