"""Measure thermodynamic state variables from OLMo-2 HuggingFace checkpoints.

Streams checkpoints one at a time: download → measure → delete → next.
Supports parallel measurement across multiple GPUs on a single node.

Usage:
    # Single GPU (sequential):
    python scripts/thermo/measure_olmo2_hf.py --model-size 7B --output /fsx/dev/jiaqi/thermo_results/olmo2_7b.jsonl

    # 8 GPUs parallel (each GPU handles 1/8 of checkpoints):
    torchrun --nproc_per_node=8 scripts/thermo/measure_olmo2_hf.py \
        --model-size 7B --output /fsx/dev/jiaqi/thermo_results/olmo2_7b.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


OLMO2_CONFIGS = {
    "1B": {
        "hf_repo": "allenai/OLMo-2-0425-1B",
        "num_params": 1_000_000_000,
        "weight_decay": 0.1,
        "batch_size": 256,
        "peak_lr": 4e-4,
    },
    "7B": {
        "hf_repo": "allenai/OLMo-2-1124-7B",
        "num_params": 7_000_000_000,
        "weight_decay": 0.1,
        "batch_size": 512,
        "peak_lr": 3e-4,
    },
    "13B": {
        "hf_repo": "allenai/OLMo-2-1124-13B",
        "num_params": 13_000_000_000,
        "weight_decay": 0.1,
        "batch_size": 512,
        "peak_lr": 3e-4,
    },
}


def discover_revisions(repo_id: str, step_interval: int = None) -> list[dict]:
    """Discover all checkpoint revisions from HuggingFace."""
    from huggingface_hub import list_repo_refs

    refs = list_repo_refs(repo_id)
    checkpoints = []
    step_pattern = re.compile(r"(?:stage\d+-)?step(\d+)(?:-tokens(\d+)B)?")

    for branch in refs.branches:
        match = step_pattern.search(branch.name)
        if not match:
            continue
        step = int(match.group(1))
        tokens_b = int(match.group(2)) if match.group(2) else None

        if step_interval and step % step_interval != 0:
            continue

        checkpoints.append({
            "revision": branch.name,
            "step": step,
            "tokens_b": tokens_b,
        })

    return sorted(checkpoints, key=lambda x: x["step"])


@torch.no_grad()
def measure_model(model, svd_k: int = 256) -> dict:
    """Compute thermodynamic state variables from a loaded model."""
    total_params = 0
    weighted_entropy = 0.0
    psi_values = []
    vol = 0.0

    for name, param in model.named_parameters():
        if param.ndim != 2:
            vol += param.data.float().pow(2).sum().item()
            total_params += param.numel()
            continue

        w = param.data.float()
        m, n = w.shape
        n_elem = m * n
        total_params += n_elem
        vol += w.pow(2).sum().item()

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
        weighted_entropy += n_elem * entropy

        # Order parameter (top-2 SVs)
        if len(sv_pos) >= 2:
            s1, s2 = sv_pos[0].item(), sv_pos[1].item()
            denom = s1 + s2
            if denom > 1e-10:
                psi_values.append((s1 - s2) / denom)

    s_global = weighted_entropy / max(total_params, 1)
    psi = np.mean(psi_values) if psi_values else 0.0

    return {
        "volume": vol,
        "spectral_entropy": s_global,
        "order_parameter": float(psi),
        "n_params": total_params,
    }


def _cleanup_hf_cache(repo_id: str, revision: str):
    """Remove cached checkpoint files to prevent disk accumulation."""
    import shutil
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    # HF caches to: HF_HOME/hub/models--{org}--{name}/snapshots/{hash}/
    repo_dir_name = f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = Path(hf_home) / "hub" / repo_dir_name / "snapshots"
    if snapshots_dir.exists():
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                shutil.rmtree(snapshot, ignore_errors=True)
    # Also clean blobs (actual weight files)
    blobs_dir = Path(hf_home) / "hub" / repo_dir_name / "blobs"
    if blobs_dir.exists():
        shutil.rmtree(blobs_dir, ignore_errors=True)
        blobs_dir.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=["1B", "7B", "13B"])
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--step-interval", type=int, default=None,
                        help="Only measure every N-th step (e.g., 5000 for sparse sampling)")
    parser.add_argument("--svd-k", type=int, default=256)
    parser.add_argument("--max-checkpoints", type=int, default=None,
                        help="Limit number of checkpoints to measure")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OLMO2_CONFIGS[args.model_size]
    repo_id = config["hf_repo"]

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        print(f"Measuring OLMo-2-{args.model_size} from {repo_id}")
        print(f"Discovering revisions...")

    # Discover checkpoints
    checkpoints = discover_revisions(repo_id, step_interval=args.step_interval)
    if args.max_checkpoints:
        checkpoints = checkpoints[:args.max_checkpoints]

    if local_rank == 0:
        print(f"Found {len(checkpoints)} checkpoints to measure")

    # Shard across GPUs
    my_checkpoints = checkpoints[local_rank::world_size]
    print(f"[GPU {local_rank}/{world_size}] Processing {len(my_checkpoints)} checkpoints")

    # Setup output
    output_path = Path(args.output)
    if world_size > 1:
        output_path = output_path.with_suffix(f".rank{local_rank}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    with open(output_path, "w") as f:
        for i, ckpt in enumerate(my_checkpoints):
            step = ckpt["step"]
            revision = ckpt["revision"]
            print(f"[GPU {local_rank}] [{i+1}/{len(my_checkpoints)}] step={step} rev={revision}")
            t0 = time.time()

            try:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    revision=revision,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True,
                )

                measurements = measure_model(model, svd_k=args.svd_k)

                record = {
                    "step": step,
                    "revision": revision,
                    "tokens_b": ckpt.get("tokens_b"),
                    "model_name": f"OLMo-2-{args.model_size}",
                    "num_params": config["num_params"],
                    "weight_decay": config["weight_decay"],
                    "batch_size": config["batch_size"],
                    **measurements,
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

                elapsed = time.time() - t0
                print(f"  S={measurements['spectral_entropy']:.4f} "
                      f"ψ={measurements['order_parameter']:.4f} "
                      f"V={measurements['volume']:.0f} "
                      f"({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                f.write(json.dumps({"step": step, "revision": revision, "error": str(e)}) + "\n")
                f.flush()

            finally:
                if "model" in locals():
                    del model
                torch.cuda.empty_cache()
                # Clean HF cache to prevent disk accumulation
                # Each 7B checkpoint is ~14GB, 970 checkpoints = 13.6TB without cleanup
                _cleanup_hf_cache(repo_id, revision)

    print(f"[GPU {local_rank}] Done. Results: {output_path}")

    # Merge sharded results on rank 0
    if world_size > 1 and local_rank == 0:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

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
