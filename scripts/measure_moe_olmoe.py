#!/usr/bin/env python3
"""Batch measurement of OLMoE-1B-7B across training checkpoints.

OLMoE has 244 checkpoints (every 5K steps up to ~1.2M steps).
This script measures a configurable subset for the spectral dynamics study.

Usage:
    # Quick test (1 checkpoint)
    python scripts/measure_moe_olmoe.py --max-ckpts 1

    # Phase 1: 10 evenly-spaced checkpoints
    python scripts/measure_moe_olmoe.py --max-ckpts 10

    # Full campaign: all 244 checkpoints (GPU recommended)
    python scripts/measure_moe_olmoe.py --device cuda

    # Sample 16 experts per layer (faster, for large-expert models)
    python scripts/measure_moe_olmoe.py --max-experts 16
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

import numpy as np


OLMOE_MODEL = "allenai/OLMoE-1B-7B-0924"
OLMOE_HIDDEN_DIM = 2048
OLMOE_NUM_EXPERTS = 64
OLMOE_TOP_K = 8

# Checkpoint revisions: OLMoE publishes checkpoints as branches
# Format: step{N}-tokens{M}B
# Known checkpoints span step 0 to step 1200000
OLMOE_CHECKPOINTS = [
    {"step": 0, "revision": "main", "tokens_b": 0},
    # We'll auto-discover available revisions via HF API
]


def discover_checkpoints(model_id: str, max_ckpts: int = 0) -> list[dict]:
    """Discover available checkpoint revisions from HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        refs = api.list_repo_refs(model_id)

        checkpoints = []
        for branch in refs.branches:
            name = branch.name
            if name == "main":
                checkpoints.append({"step": 1200000, "revision": "main", "tokens_b": 5000})
                continue

            # Parse step{N}-tokens{M}B format
            import re
            m = re.match(r"step(\d+)-tokens(\d+)B", name)
            if m:
                step = int(m.group(1))
                tokens_b = int(m.group(2))
                checkpoints.append({"step": step, "revision": name, "tokens_b": tokens_b})

        checkpoints.sort(key=lambda c: c["step"])
        print(f"Discovered {len(checkpoints)} checkpoints for {model_id}")

        if max_ckpts > 0 and len(checkpoints) > max_ckpts:
            # Evenly sample, always including first and last
            indices = np.linspace(0, len(checkpoints) - 1, max_ckpts, dtype=int)
            checkpoints = [checkpoints[i] for i in indices]
            print(f"Sampled {len(checkpoints)} checkpoints")

        return checkpoints

    except ImportError:
        print("huggingface_hub not available, using manual checkpoint list")
        return _manual_checkpoints(max_ckpts)
    except Exception as e:
        print(f"Failed to discover checkpoints: {e}")
        return _manual_checkpoints(max_ckpts)


def _manual_checkpoints(max_ckpts: int) -> list[dict]:
    """Fallback: manually specified key checkpoints."""
    ckpts = [
        {"step": 0, "revision": "step0-tokens0B", "tokens_b": 0},
        {"step": 5000, "revision": "step5000-tokens21B", "tokens_b": 21},
        {"step": 10000, "revision": "step10000-tokens42B", "tokens_b": 42},
        {"step": 50000, "revision": "step50000-tokens210B", "tokens_b": 210},
        {"step": 100000, "revision": "step100000-tokens419B", "tokens_b": 419},
        {"step": 200000, "revision": "step200000-tokens839B", "tokens_b": 839},
        {"step": 400000, "revision": "step400000-tokens1678B", "tokens_b": 1678},
        {"step": 600000, "revision": "step600000-tokens2516B", "tokens_b": 2516},
        {"step": 800000, "revision": "step800000-tokens3355B", "tokens_b": 3355},
        {"step": 1000000, "revision": "step1000000-tokens4194B", "tokens_b": 4194},
        {"step": 1200000, "revision": "main", "tokens_b": 5033},
    ]
    if max_ckpts > 0 and len(ckpts) > max_ckpts:
        indices = np.linspace(0, len(ckpts) - 1, max_ckpts, dtype=int)
        ckpts = [ckpts[i] for i in indices]
    return ckpts


def main():
    parser = argparse.ArgumentParser(description="Measure OLMoE spectral dynamics")
    parser.add_argument("--max-ckpts", type=int, default=10, help="Max checkpoints to measure (0=all)")
    parser.add_argument("--max-experts", type=int, default=0, help="Max experts per layer (0=all 64)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--output-dir", default="results/olmoe_moe", help="Output directory")
    parser.add_argument("--no-alignment", action="store_true", help="Skip alignment computation")
    parser.add_argument("--resume", action="store_true", help="Skip already-measured checkpoints")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "olmoe_1b_7b.jsonl"

    # Discover checkpoints
    checkpoints = discover_checkpoints(OLMOE_MODEL, max_ckpts=args.max_ckpts)

    # Check what's already measured
    measured_steps = set()
    if args.resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    measured_steps.add(d["step"])
                except Exception:
                    pass
        print(f"Already measured {len(measured_steps)} checkpoints")

    from experiments.thermodynamics.moe_measures import measure_moe_checkpoint

    for i, ckpt in enumerate(checkpoints):
        if ckpt["step"] in measured_steps:
            print(f"[{i+1}/{len(checkpoints)}] Step {ckpt['step']}: already measured, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(checkpoints)}] Step {ckpt['step']} ({ckpt['tokens_b']}B tokens)")
        print(f"{'='*60}")

        try:
            measure_moe_checkpoint(
                model_name_or_path=OLMOE_MODEL,
                revision=ckpt["revision"],
                step=ckpt["step"],
                hidden_dim=OLMOE_HIDDEN_DIM,
                num_experts=OLMOE_NUM_EXPERTS,
                top_k_routing=OLMOE_TOP_K,
                device=args.device,
                max_experts_per_layer=args.max_experts,
                measure_alignment=not args.no_alignment,
                output_path=str(output_file),
            )
        except Exception as e:
            print(f"ERROR measuring step {ckpt['step']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone. Results saved to {output_file}")


if __name__ == "__main__":
    main()
