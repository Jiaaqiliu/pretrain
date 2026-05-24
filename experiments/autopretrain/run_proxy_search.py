"""Run proxy-scale data mixture search (Option B, Phase 1).

This script orchestrates the MCGS search at 190M scale:
- 50 cycles, each training for 5000 steps (~5B tokens)
- Search over data mix ratios, filter aggressiveness, and curriculum
- Evaluates on C4 val loss + per-domain decomposition
- Outputs ranked recipe list for Phase 2 transfer verification

Usage:
    # Full search (requires H200 cluster)
    python experiments/autopretrain/run_proxy_search.py

    # Mock mode (tests the search loop locally, no GPU needed)
    python experiments/autopretrain/run_proxy_search.py --mock

    # Fewer cycles for development
    python experiments/autopretrain/run_proxy_search.py --mock --cycles 5

Expected compute: ~100 GPU-hours on 8x H200 (50 cycles × ~2 GPU-hours each)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_evolve.model.algorithms.autopretrain.search import (
    ProxySearchLoop,
    SearchConfig,
)
from agent_evolve.model.algorithms.autopretrain.mutator import MixtureMutator
from agent_evolve.model.algorithms.autopretrain.reward import PretrainRewardPolicy
from agent_evolve.model.algorithms.autopretrain.eval_harness import PretrainEvalHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run proxy-scale data mixture search")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no GPU)")
    parser.add_argument("--cycles", type=int, default=50, help="Number of MCGS cycles")
    parser.add_argument("--steps-per-trial", type=int, default=5000, help="Training steps per trial")
    parser.add_argument("--model", type=str, default="olmo2_190M", help="Proxy model factory")
    parser.add_argument("--output-dir", type=str, default="./results/autopretrain_search", help="Output directory")
    parser.add_argument("--initial-mix", type=str, default="olmo2_stage1", help="Starting mixture reference")
    parser.add_argument("--data-root", type=str, default="/fsx/dev/jiaqi/data/olmo-3b-pretrain")
    parser.add_argument("--gpus", type=int, default=8, help="GPUs to use")
    parser.add_argument("--exploration-c", type=float, default=1.4, help="UCT exploration constant")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AutoPretrain: Proxy-Scale Data Mixture Search")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Cycles: {args.cycles}")
    logger.info(f"Steps/trial: {args.steps_per_trial}")
    logger.info(f"Initial mix: {args.initial_mix}")
    logger.info(f"Mock mode: {args.mock}")
    logger.info("=" * 60)

    config = SearchConfig(
        max_cycles=args.cycles,
        exploration_c=args.exploration_c,
        proxy_model_factory=args.model,
        proxy_steps_per_trial=args.steps_per_trial,
        proxy_batch_size=64 if args.model == "olmo2_190M" else 32,
        proxy_gpus=args.gpus,
        data_root=args.data_root,
        fast_eval=True,
        output_dir=args.output_dir,
        initial_mixture=args.initial_mix,
    )

    mutator = MixtureMutator(perturbation_scale=0.08)
    reward_policy = PretrainRewardPolicy()
    eval_harness = PretrainEvalHarness(fast_mode=True)

    loop = ProxySearchLoop(
        config=config,
        mutator=mutator,
        reward_policy=reward_policy,
        eval_harness=eval_harness,
    )

    result = loop.run()

    # Print summary
    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    # Save final results
    output_path = Path(args.output_dir) / "final_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    logger.info("Results saved to %s", output_path)

    # Print top recipes in a format ready for Phase 2
    print("\n\nTop recipes for Phase 2 transfer verification:")
    print("-" * 40)
    for i, node in enumerate(result.top_k(5)):
        print(f"\n#{i+1}: {node.node_id}")
        print(f"  Val loss: {node.eval_result.val_loss:.4f}")
        print(f"  Curriculum: {node.curriculum}")
        if node.mutation_record:
            print(f"  Found via: {node.mutation_record.description}")


if __name__ == "__main__":
    main()
