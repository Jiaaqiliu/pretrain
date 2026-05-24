"""Phase 2: Transfer Verification — Run top recipes at 3B scale.

Takes the top-K recipes discovered by the proxy search (190M) and trains
each at 3B scale for 10K steps. Measures rank correlation between proxy
and target performance to validate the transfer hypothesis.

Usage:
    python experiments/autopretrain/run_transfer_verification.py \
        --recipes results/phase1_proxy/final_results.json \
        --model olmo2_3B \
        --steps 10000 \
        --top-k 5 \
        --output-dir results/phase2_transfer

Expected compute: ~500 GPU-hours (5 recipes × 100 GPU-hours each on 16 GPUs)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_evolve.model.algorithms.autopretrain.mixture import CurriculumSchedule, DataMixture
from agent_evolve.model.algorithms.autopretrain.workspace_bridge import WorkspaceBridge
from agent_evolve.model.algorithms.autopretrain.eval_harness import PretrainEvalHarness, EvalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer verification at 3B scale")
    parser.add_argument("--recipes", type=str, required=True, help="Path to Phase 1 results JSON")
    parser.add_argument("--model", type=str, default="olmo2_3B", help="Target model factory")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps per recipe")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top recipes to verify")
    parser.add_argument("--output-dir", type=str, default="./results/phase2_transfer")
    parser.add_argument("--data-root", type=str, default="/fsx/dev/jiaqi/data/olmo-3b-pretrain")
    parser.add_argument("--gpus", type=int, default=16, help="GPUs (typically 2 nodes × 8)")
    parser.add_argument("--include-baselines", action="store_true", help="Also run OLMo-2 and Llama-3 baselines")
    return parser.parse_args()


def load_recipes(recipes_path: str, top_k: int) -> list[dict[str, Any]]:
    """Load top-K recipes from Phase 1 results."""
    data = json.loads(Path(recipes_path).read_text())
    top_nodes = data.get("top_5", data.get("ranked_nodes", []))[:top_k]
    return top_nodes


def recipe_to_curriculum(recipe: dict) -> CurriculumSchedule:
    """Convert a recipe dict back into a CurriculumSchedule."""
    if "curriculum" in recipe:
        return CurriculumSchedule.from_dict(recipe["curriculum"])
    elif "mixture" in recipe:
        mixture = DataMixture.from_dict(recipe["mixture"])
        return CurriculumSchedule.static(mixture)
    else:
        return CurriculumSchedule.static(DataMixture.from_reference("olmo2_stage1"))


def get_baselines() -> list[tuple[str, CurriculumSchedule]]:
    """Return baseline curricula for comparison."""
    baselines = [
        ("baseline_olmo2_stage1", CurriculumSchedule.static(
            DataMixture.from_reference("olmo2_stage1")
        )),
        ("baseline_llama3", CurriculumSchedule.static(
            DataMixture.from_reference("llama3")
        )),
        ("baseline_olmo2_midtrain", CurriculumSchedule.static(
            DataMixture.from_reference("olmo2_midtrain")
        )),
    ]
    return baselines


def compute_rank_correlation(proxy_losses: list[float], target_losses: list[float]) -> float:
    """Compute Spearman rank correlation between proxy and target rankings."""
    n = len(proxy_losses)
    if n < 3:
        return 0.0

    # Rank both lists
    def rank(vals):
        sorted_indices = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = rank_val + 1
        return ranks

    proxy_ranks = rank(proxy_losses)
    target_ranks = rank(target_losses)

    # Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
    d_sq_sum = sum((p - t) ** 2 for p, t in zip(proxy_ranks, target_ranks))
    rho = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))
    return rho


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 2: Transfer Verification")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Steps: {args.steps}")
    logger.info(f"  Top-K: {args.top_k}")
    logger.info(f"  GPUs: {args.gpus}")
    logger.info("=" * 60)

    # Load Phase 1 results
    recipes = load_recipes(args.recipes, args.top_k)
    logger.info(f"Loaded {len(recipes)} recipes from Phase 1")

    # Build experiment list
    experiments: list[tuple[str, CurriculumSchedule, float]] = []

    for i, recipe in enumerate(recipes):
        name = recipe.get("node_id", f"recipe_{i}")
        curriculum = recipe_to_curriculum(recipe)
        proxy_loss = recipe.get("eval_result", {}).get("val_loss", 4.0)
        experiments.append((name, curriculum, proxy_loss))
        logger.info(f"  Recipe {i+1}: {name} (proxy loss={proxy_loss:.4f})")

    if args.include_baselines:
        for name, curriculum in get_baselines():
            experiments.append((name, curriculum, 4.0))  # No proxy loss for baselines
            logger.info(f"  Baseline: {name}")

    # Create workspace bridge for 3B model
    bridge = WorkspaceBridge(
        trials_root=output_dir / "trials",
        data_root=args.data_root,
    )

    # Run each experiment
    results: list[dict[str, Any]] = []

    for name, curriculum, proxy_loss in experiments:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running: {name}")
        logger.info(f"  Curriculum: {curriculum}")

        trial_dir = bridge.create_trial_workspace(
            node_id=f"transfer_{name}",
            curriculum=curriculum,
            model_factory=args.model,
            max_steps=args.steps,
            batch_size=256,  # Full batch for 3B
            sequence_length=4096,
            lr=3e-4,
            warmup_steps=1000,
        )

        logger.info(f"  Workspace: {trial_dir}")
        logger.info(f"  Launch training via: torchrun --nproc-per-node={args.gpus} ...")

        # In production, this would call OLMoCoreBackend.run_trial()
        # For now, save the workspace and let the user launch manually
        results.append({
            "name": name,
            "curriculum": curriculum.to_dict(),
            "proxy_loss": proxy_loss,
            "workspace_dir": str(trial_dir),
            "target_model": args.model,
            "target_steps": args.steps,
            "status": "workspace_ready",
        })

    # Save experiment plan
    plan_path = output_dir / "transfer_plan.json"
    plan_path.write_text(json.dumps({
        "experiments": results,
        "config": {
            "model": args.model,
            "steps": args.steps,
            "gpus": args.gpus,
            "data_root": args.data_root,
        },
    }, indent=2, default=str))

    logger.info(f"\n{'='*60}")
    logger.info(f"Transfer verification plan saved to {plan_path}")
    logger.info(f"{len(results)} experiments ready to launch")
    logger.info("=" * 60)

    # If we have both proxy and target losses, compute correlation
    proxy_losses = [r["proxy_loss"] for r in results if r["proxy_loss"] < 4.0]
    if len(proxy_losses) >= 3:
        logger.info(f"\nProxy losses: {proxy_losses}")
        logger.info("After target training completes, run analyze_transfer.py to compute Spearman ρ")


if __name__ == "__main__":
    main()
