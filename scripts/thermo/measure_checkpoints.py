"""Batch measurement of thermodynamic state variables from OLMo checkpoints.

Usage (single GPU, local):
    python scripts/thermo/measure_checkpoints.py \
        --model-size 190M \
        --checkpoint-dir /fsx/dev/jiaqi/checkpoints/olmo-190m \
        --output /fsx/dev/jiaqi/thermo_results/measurements_190m.jsonl

Usage (K8s job — see scripts/k8s/thermo/thermo_measure_*.yaml):
    torchrun --nproc_per_node=1 scripts/thermo/measure_checkpoints.py \
        --model-size 7B --checkpoint-dir ... --output ...

Each checkpoint is processed independently (embarrassingly parallel).
When run with torchrun, checkpoints are sharded across GPUs.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.thermodynamics.checkpoint_loader import (
    CheckpointInfo,
    OLMO_CONFIGS,
    discover_checkpoints,
    discover_hf_checkpoints,
    load_olmo_checkpoint,
    load_training_logs,
)
from experiments.thermodynamics.measures import measure_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Measure thermodynamic state variables from OLMo checkpoints")
    parser.add_argument("--model-size", required=True, choices=["190M", "1B", "7B", "13B"])
    parser.add_argument("--checkpoint-dir", type=str, help="Local checkpoint directory")
    parser.add_argument("--use-hf", action="store_true", help="Load from HuggingFace Hub instead")
    parser.add_argument("--training-log", type=str, help="Path to training log (JSON/CSV/JSONL)")
    parser.add_argument("--output", required=True, type=str, help="Output JSONL path")
    parser.add_argument("--step-range", type=str, default=None, help="Step range: min,max")
    parser.add_argument("--step-interval", type=int, default=None, help="Process every N-th checkpoint")
    parser.add_argument("--svd-k", type=int, default=256, help="Top-k singular values for randomized SVD")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--default-grad-variance", type=float, default=1e-6,
                        help="Fallback gradient variance when not available from logs")
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    config = OLMO_CONFIGS[args.model_size]
    weight_decay = config["weight_decay"]
    batch_size = config["batch_size"]

    # Distributed setup (for multi-GPU parallel processing)
    local_rank = 0
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    elif "LOCAL_RANK" in __import__("os").environ:
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        world_size = int(__import__("os").environ.get("WORLD_SIZE", 1))

    # Discover checkpoints
    step_range = None
    if args.step_range:
        parts = args.step_range.split(",")
        step_range = (int(parts[0]), int(parts[1]))

    if args.use_hf:
        checkpoints = discover_hf_checkpoints(args.model_size)
    else:
        if not args.checkpoint_dir:
            print("ERROR: --checkpoint-dir required when not using --use-hf")
            sys.exit(1)
        checkpoints = discover_checkpoints(
            args.checkpoint_dir,
            args.model_size,
            step_range=step_range,
            step_interval=args.step_interval,
        )

    if not checkpoints:
        print(f"No checkpoints found for {args.model_size}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints for OLMo-{args.model_size}")

    # Shard checkpoints across ranks
    my_checkpoints = checkpoints[local_rank::world_size]
    print(f"Rank {local_rank}/{world_size}: processing {len(my_checkpoints)} checkpoints")

    # Load training logs for LR and gradient stats
    training_logs = {}
    if args.training_log:
        training_logs = load_training_logs(args.training_log)
        print(f"Loaded training logs: {len(training_logs)} entries")

    # Process each checkpoint
    output_path = Path(args.output)
    if world_size > 1:
        output_path = output_path.with_suffix(f".rank{local_rank}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, ckpt_info in enumerate(my_checkpoints):
            step = ckpt_info.step
            print(f"[{i+1}/{len(my_checkpoints)}] Processing step {step}...")
            t0 = time.time()

            try:
                model = load_olmo_checkpoint(ckpt_info, device=args.device, dtype=dtype)

                # Get training metadata from logs or checkpoint info
                log_entry = training_logs.get(step, {})
                lr = ckpt_info.lr_at_step or log_entry.get("lr", log_entry.get("learning_rate", 0.0))
                loss = ckpt_info.loss_at_step or log_entry.get("loss", log_entry.get("train_loss", 0.0))
                grad_var = log_entry.get("grad_variance", log_entry.get("grad_var", args.default_grad_variance))

                if lr is None or lr == 0.0:
                    lr = 0.0
                    print(f"  WARNING: no LR found for step {step}, using 0.0")

                measurement = measure_checkpoint(
                    model=model,
                    step=step,
                    loss=float(loss),
                    lr=float(lr),
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    grad_variance=float(grad_var),
                    model_name=f"OLMo-{args.model_size}",
                    k=args.svd_k,
                    checkpoint_path=ckpt_info.path,
                )

                # Serialize (skip layer_measurements for compact output)
                record = asdict(measurement)
                record.pop("layer_measurements", None)
                f.write(json.dumps(record) + "\n")
                f.flush()

                elapsed = time.time() - t0
                print(f"  Done: S={measurement.spectral_entropy:.4f}, "
                      f"ψ={measurement.order_parameter:.4f}, "
                      f"F={measurement.free_energy:.4f}, "
                      f"PV/NT={measurement.pv_over_nt:.4f} "
                      f"({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR processing step {step}: {e}")
                error_record = {"step": step, "error": str(e), "model_name": f"OLMo-{args.model_size}"}
                f.write(json.dumps(error_record) + "\n")
                f.flush()

            finally:
                # Free GPU memory
                if "model" in dir():
                    del model
                torch.cuda.empty_cache()

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
