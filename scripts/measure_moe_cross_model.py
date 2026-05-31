#!/usr/bin/env python3
"""Cross-model MoE spectral measurement campaign.

Measures final checkpoints of multiple MoE models to test:
  - H5: Phase transition threshold (total vs active vs per-expert params)
  - H6: MLP bottleneck asymmetry in MoE
  - Architecture comparison (shared expert vs pure MoE)

Usage:
    # Tier 1: Small models (CPU feasible)
    python scripts/measure_moe_cross_model.py --tier 1

    # Tier 2: Medium models (GPU recommended)
    python scripts/measure_moe_cross_model.py --tier 2 --device cuda

    # Single model
    python scripts/measure_moe_cross_model.py --model mistralai/Mixtral-8x7B-v0.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path


# Model registry with metadata
MOE_MODELS = {
    # Tier 1: Small / manageable on CPU (< 50B total params)
    1: [
        {
            "model_id": "allenai/OLMoE-1B-7B-0924",
            "name": "OLMoE-1B-7B",
            "total_params_b": 6.9,
            "active_params_b": 1.3,
            "num_experts": 64,
            "top_k": 8,
            "hidden_dim": 2048,
            "has_shared": False,
            "arch_family": "OLMo",
        },
        {
            "model_id": "microsoft/Phi-3.5-MoE-instruct",
            "name": "Phi-3.5-MoE",
            "total_params_b": 42,
            "active_params_b": 6.6,
            "num_experts": 16,
            "top_k": 2,
            "hidden_dim": 4096,
            "has_shared": False,
            "arch_family": "Phi",
        },
        {
            "model_id": "mistralai/Mixtral-8x7B-v0.1",
            "name": "Mixtral-8x7B",
            "total_params_b": 46.7,
            "active_params_b": 12.9,
            "num_experts": 8,
            "top_k": 2,
            "hidden_dim": 4096,
            "has_shared": False,
            "arch_family": "Mistral",
        },
    ],

    # Tier 2: Medium models (GPU needed, < 200B)
    2: [
        {
            "model_id": "mistralai/Mixtral-8x22B-v0.1",
            "name": "Mixtral-8x22B",
            "total_params_b": 141,
            "active_params_b": 39,
            "num_experts": 8,
            "top_k": 2,
            "hidden_dim": 6144,
            "has_shared": False,
            "arch_family": "Mistral",
        },
        {
            "model_id": "databricks/dbrx-base",
            "name": "DBRX",
            "total_params_b": 132,
            "active_params_b": 36,
            "num_experts": 16,
            "top_k": 4,
            "hidden_dim": 6144,
            "has_shared": False,
            "arch_family": "DBRX",
        },
        {
            "model_id": "Qwen/Qwen2-57B-A14B",
            "name": "Qwen2-MoE-57B",
            "total_params_b": 57,
            "active_params_b": 14,
            "num_experts": 64,
            "top_k": 8,
            "hidden_dim": 3584,
            "has_shared": True,
            "arch_family": "Qwen",
        },
    ],

    # Tier 3: Large models (multi-GPU, > 200B)
    3: [
        {
            "model_id": "deepseek-ai/DeepSeek-V2",
            "name": "DeepSeek-V2",
            "total_params_b": 236,
            "active_params_b": 21,
            "num_experts": 160,
            "top_k": 6,
            "hidden_dim": 5120,
            "has_shared": True,
            "arch_family": "DeepSeek",
        },
        {
            "model_id": "deepseek-ai/DeepSeek-V3",
            "name": "DeepSeek-V3",
            "total_params_b": 671,
            "active_params_b": 37,
            "num_experts": 256,
            "top_k": 8,
            "hidden_dim": 7168,
            "has_shared": True,
            "arch_family": "DeepSeek",
        },
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Cross-model MoE spectral measurement")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3], help="Model tier (1=small, 2=medium, 3=large)")
    parser.add_argument("--model", default=None, help="Measure a single model by HF ID")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max-experts", type=int, default=0, help="Max experts per layer to measure")
    parser.add_argument("--output-dir", default="results/moe_cross_model", help="Output directory")
    parser.add_argument("--no-alignment", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cross_model_moe.jsonl"

    from experiments.thermodynamics.moe_measures import measure_moe_checkpoint

    if args.model:
        # Single model mode
        models = [{"model_id": args.model, "name": args.model.split("/")[-1]}]
    else:
        # Tier mode: include all tiers up to and including the specified one
        models = []
        for t in range(1, args.tier + 1):
            models.extend(MOE_MODELS[t])

    for i, m in enumerate(models):
        print(f"\n{'#'*60}")
        print(f"[{i+1}/{len(models)}] {m.get('name', m['model_id'])}")
        print(f"  Total: {m.get('total_params_b', '?')}B, Active: {m.get('active_params_b', '?')}B")
        print(f"  Experts: {m.get('num_experts', '?')}, top-{m.get('top_k', '?')}")
        print(f"{'#'*60}")

        try:
            measure_moe_checkpoint(
                model_name_or_path=m["model_id"],
                hidden_dim=m.get("hidden_dim", 0),
                num_experts=m.get("num_experts", 0),
                top_k_routing=m.get("top_k", 0),
                device=args.device,
                max_experts_per_layer=args.max_experts,
                measure_alignment=not args.no_alignment,
                output_path=str(output_file),
            )
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll done. Results saved to {output_file}")


if __name__ == "__main__":
    main()
