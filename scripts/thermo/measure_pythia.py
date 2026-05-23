"""Measure thermodynamic state variables from Pythia checkpoints.

Streams checkpoints from HuggingFace: download → measure → delete → next.
Supports parallel measurement across multiple GPUs on a single node.

Usage:
    # Single GPU:
    python scripts/thermo/measure_pythia.py --model-size 70m --output results/pythia/pythia_70m.jsonl

    # 8 GPUs parallel:
    torchrun --nproc_per_node=8 scripts/thermo/measure_pythia.py \
        --model-size 70m --output results/pythia/pythia_70m.jsonl
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

# Pythia training configurations (from paper: Biderman et al., ICML 2023)
PYTHIA_CONFIGS = {
    "70m": {
        "hf_repo": "EleutherAI/pythia-70m-deduped",
        "num_params": 70_400_000,
        "num_layers": 6,
        "hidden_dim": 512,
        "num_heads": 8,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
        "peak_lr": 1.0e-3,
        "min_lr": 1.0e-4,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "total_tokens_b": 300,
    },
    "160m": {
        "hf_repo": "EleutherAI/pythia-160m-deduped",
        "num_params": 162_300_000,
        "num_layers": 12,
        "hidden_dim": 768,
        "num_heads": 12,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
        "peak_lr": 6.0e-4,
        "min_lr": 6.0e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "total_tokens_b": 300,
    },
    "410m": {
        "hf_repo": "EleutherAI/pythia-410m-deduped",
        "num_params": 405_300_000,
        "num_layers": 24,
        "hidden_dim": 1024,
        "num_heads": 16,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
        "peak_lr": 3.0e-4,
        "min_lr": 3.0e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "total_tokens_b": 300,
    },
    "1b": {
        "hf_repo": "EleutherAI/pythia-1b-deduped",
        "num_params": 1_011_800_000,
        "num_layers": 16,
        "hidden_dim": 2048,
        "num_heads": 8,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
        "peak_lr": 2.5e-4,
        "min_lr": 2.5e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "total_tokens_b": 300,
    },
    "1.4b": {
        "hf_repo": "EleutherAI/pythia-1.4b-deduped",
        "num_params": 1_414_600_000,
        "num_layers": 24,
        "hidden_dim": 2048,
        "num_heads": 16,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
        "peak_lr": 2.0e-4,
        "min_lr": 2.0e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "total_tokens_b": 300,
    },
    "2.8b": {
        "hf_repo": "EleutherAI/pythia-2.8b-deduped",
        "num_params": 2_775_200_000,
        "num_layers": 32,
        "hidden_dim": 2560,
        "num_heads": 32,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
        "peak_lr": 1.6e-4,
        "min_lr": 1.6e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "total_tokens_b": 300,
    },
    "6.9b": {
        "hf_repo": "EleutherAI/pythia-6.9b-deduped",
        "num_params": 6_857_300_000,
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
        "peak_lr": 1.2e-4,
        "min_lr": 1.2e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "total_tokens_b": 300,
    },
    "12b": {
        "hf_repo": "EleutherAI/pythia-12b-deduped",
        "num_params": 11_846_100_000,
        "num_layers": 36,
        "hidden_dim": 5120,
        "num_heads": 40,
        "weight_decay": 0.1,
        "batch_size_tokens": 2_097_152,
        "peak_lr": 1.2e-4,
        "min_lr": 1.2e-5,
        "total_steps": 143_000,
        "warmup_steps": 1_430,
        "total_tokens_b": 300,
    },
}

# Checkpoint steps to measure (25 key points covering full training)
SAMPLE_STEPS = [
    # Log-spaced early phase (11 checkpoints)
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    # Main training at 10K intervals (14 checkpoints)
    1000, 10000, 20000, 30000, 40000, 50000, 60000,
    70000, 80000, 90000, 100000, 110000, 120000, 143000,
]

# Dense sampling option (every 1000 steps, 154 total)
ALL_STEPS = (
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    + list(range(1000, 143001, 1000))
)


def cosine_lr_at_step(step: int, config: dict) -> float:
    """Compute the cosine LR at a given step (Pythia uses cosine schedule)."""
    warmup = config["warmup_steps"]
    total = config["total_steps"]
    max_lr = config["peak_lr"]
    min_lr = config["min_lr"]

    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


@torch.no_grad()
def measure_model(model, svd_k: int = 256) -> dict:
    """Compute thermodynamic state variables from a loaded model."""
    total_2d_params = 0
    total_all_params = 0
    weighted_entropy = 0.0
    psi_values = []
    vol = 0.0

    for name, param in model.named_parameters():
        n_elem = param.numel()
        total_all_params += n_elem
        vol += param.data.float().pow(2).sum().item()

        if param.ndim != 2:
            continue

        w = param.data.float()
        m, n = w.shape
        total_2d_params += m * n

        min_dim = min(m, n)
        if min_dim < 2:
            continue

        # SVD
        if min_dim <= 2048:
            sv = torch.linalg.svdvals(w)
        else:
            actual_k = min(svd_k, min_dim)
            omega = torch.randn(n, actual_k + 16, device=w.device, dtype=torch.float32)
            y = w @ omega
            q, _ = torch.linalg.qr(y)
            z = w.T @ q
            q, _ = torch.linalg.qr(w @ z)
            b = q.T @ w
            sv = torch.linalg.svdvals(b)[:actual_k]

        sv_pos = sv[sv > 0]
        if len(sv_pos) == 0:
            continue

        # Spectral entropy
        p = sv_pos / sv_pos.sum()
        entropy = -(p * torch.log(p)).sum().item()
        weighted_entropy += (m * n) * entropy

        # Order parameter (top-2 SVs)
        if len(sv_pos) >= 2:
            s1, s2 = sv_pos[0].item(), sv_pos[1].item()
            denom = s1 + s2
            if denom > 1e-10:
                psi_values.append((s1 - s2) / denom)

    s_global = weighted_entropy / max(total_2d_params, 1)
    psi = float(np.mean(psi_values)) if psi_values else 0.0

    return {
        "volume": vol,
        "spectral_entropy": s_global,
        "order_parameter": float(psi),
        "n_params_total": total_all_params,
        "n_params_2d": total_2d_params,
        "n_layers_measured": len(psi_values),
    }


def cleanup_hf_cache(repo_id: str):
    """Remove cached checkpoint files to prevent disk accumulation."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    repo_dir_name = f"models--{repo_id.replace('/', '--')}"
    repo_cache = Path(hf_home) / "hub" / repo_dir_name
    if repo_cache.exists():
        shutil.rmtree(repo_cache, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Pythia checkpoints")
    parser.add_argument("--model-size", required=True,
                        choices=list(PYTHIA_CONFIGS.keys()))
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--svd-k", type=int, default=256)
    parser.add_argument("--dense", action="store_true",
                        help="Measure all 154 checkpoints instead of sampled 25")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated list of specific steps to measure")
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps already in output file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PYTHIA_CONFIGS[args.model_size]
    repo_id = config["hf_repo"]

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Per-GPU HF cache to avoid race conditions
    base_hf_home = os.environ.get("HF_HOME", "/fsx/dev/jiaqi/.cache/huggingface")
    per_gpu_hf_home = f"{base_hf_home}_rank{local_rank}"
    os.environ["HF_HOME"] = per_gpu_hf_home
    Path(per_gpu_hf_home).mkdir(parents=True, exist_ok=True)

    # Determine which steps to measure
    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    elif args.dense:
        steps = ALL_STEPS
    else:
        steps = SAMPLE_STEPS

    # Resume support
    completed_steps = set()
    output_path = Path(args.output)
    if world_size > 1:
        output_path = output_path.with_suffix(f".rank{local_rank}.jsonl")

    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                record = json.loads(line)
                if "error" not in record:
                    completed_steps.add(record["step"])
        if local_rank == 0:
            print(f"Resuming: {len(completed_steps)} steps already completed")

    steps = [s for s in steps if s not in completed_steps]

    if local_rank == 0:
        print(f"Measuring Pythia-{args.model_size} ({repo_id})")
        print(f"Steps to measure: {len(steps)} ({'dense' if args.dense else 'sampled'})")
        print(f"World size: {world_size}, Per-GPU cache: {per_gpu_hf_home}")

    # Shard across GPUs
    my_steps = steps[local_rank::world_size]
    print(f"[GPU {local_rank}/{world_size}] Processing {len(my_steps)} checkpoints")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    mode = "a" if args.resume else "w"
    with open(output_path, mode) as f:
        for i, step in enumerate(my_steps):
            revision = f"step{step}"
            lr = cosine_lr_at_step(step, config)
            tokens_b = step * config["batch_size_tokens"] / 1e9

            print(f"[GPU {local_rank}] [{i+1}/{len(my_steps)}] step={step} rev={revision}")
            t0 = time.time()

            try:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    revision=revision,
                    torch_dtype=torch.float32,
                    device_map=device,
                    trust_remote_code=True,
                )

                measurements = measure_model(model, svd_k=args.svd_k)

                # Compute PV/(NT) - state equation variable
                V = measurements["volume"]
                N = measurements["n_params_total"]
                T_proxy = lr
                P = config["weight_decay"]
                pv_over_nt = (P * V) / (N * T_proxy) if T_proxy > 1e-15 else 0.0

                record = {
                    "step": step,
                    "revision": revision,
                    "tokens_b": round(tokens_b, 2),
                    "model_name": f"pythia-{args.model_size}-deduped",
                    "model_size": args.model_size,
                    "num_params": config["num_params"],
                    "hidden_dim": config["hidden_dim"],
                    "num_layers": config["num_layers"],
                    "weight_decay": config["weight_decay"],
                    "batch_size_tokens": config["batch_size_tokens"],
                    "lr": lr,
                    "peak_lr": config["peak_lr"],
                    "pv_over_nt": pv_over_nt,
                    **measurements,
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

                elapsed = time.time() - t0
                print(f"  S={measurements['spectral_entropy']:.4f} "
                      f"ψ={measurements['order_parameter']:.4f} "
                      f"V={measurements['volume']:.0f} "
                      f"PV/NT={pv_over_nt:.4f} "
                      f"({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                record = {"step": step, "revision": revision, "error": str(e)}
                f.write(json.dumps(record) + "\n")
                f.flush()

            finally:
                if "model" in locals():
                    del model
                torch.cuda.empty_cache()
                cleanup_hf_cache(repo_id)

    print(f"[GPU {local_rank}] Done. Results: {output_path}")

    # Merge sharded results on rank 0
    if world_size > 1 and local_rank == 0:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

        import time as _t
        _t.sleep(10)  # Wait for other ranks to finish writing

        merged_path = Path(args.output)
        all_records = []
        for rank in range(world_size):
            shard_path = merged_path.with_suffix(f".rank{rank}.jsonl")
            if shard_path.exists():
                with open(shard_path) as sf:
                    for line in sf:
                        all_records.append(json.loads(line))

        all_records.sort(key=lambda r: r.get("step", 0))
        with open(merged_path, "w") as mf:
            for r in all_records:
                mf.write(json.dumps(r) + "\n")
        print(f"Merged {len(all_records)} records to {merged_path}")


if __name__ == "__main__":
    main()
