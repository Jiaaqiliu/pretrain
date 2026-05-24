"""Experiment: Does the Bitter Lesson hold for reasoning tasks?

Tests whether less filtering + more data beats aggressive filtering for
reasoning-specific benchmarks (unlike the Stanford paper's general LM finding).

Design:
  - Fix data mixture to OLMo-2 ratios
  - Vary filter retention rate: [5%, 20%, 50%, 80%, 100%]
  - Train at both 190M and 3B scale
  - Evaluate on general LM (C4 loss) AND reasoning (ARC, GSM8K)
  - Compare: does the crossover happen later for reasoning?

Usage:
    python experiments/autopretrain/run_bitter_lesson.py \
        --filter-levels 0.05,0.20,0.50,0.80,1.00 \
        --model olmo2_190M \
        --steps 5000

Expected outcome: For general LM, less filtering is better at larger scale
(confirming Stanford). For reasoning, aggressive filtering helps longer.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_evolve.model.algorithms.autopretrain.mixture import (
    CurriculumSchedule,
    DataMixture,
    FilterConfig,
    REFERENCE_MIXTURES,
)
from agent_evolve.model.algorithms.autopretrain.workspace_bridge import WorkspaceBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Bitter Lesson experiment for reasoning")
    parser.add_argument("--filter-levels", type=str, default="0.05,0.20,0.50,0.80,1.00",
                        help="Comma-separated filter retention rates to test")
    parser.add_argument("--model", type=str, default="olmo2_190M")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="./results/bitter_lesson")
    parser.add_argument("--data-root", type=str, default="/fsx/dev/jiaqi/data/olmo-3b-pretrain")
    parser.add_argument("--base-mix", type=str, default="olmo2_stage1", help="Base mixture to use")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_levels = [float(x) for x in args.filter_levels.split(",")]

    logger.info("=" * 60)
    logger.info("Bitter Lesson Experiment: Filter Aggressiveness vs Reasoning")
    logger.info(f"  Filter levels: {filter_levels}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Steps: {args.steps}")
    logger.info(f"  Base mix: {args.base_mix}")
    logger.info("=" * 60)

    bridge = WorkspaceBridge(
        trials_root=output_dir / "trials",
        data_root=args.data_root,
    )

    base_weights = dict(REFERENCE_MIXTURES[args.base_mix])
    experiments = []

    for retention in filter_levels:
        # Apply uniform filter retention across all domains
        filter_config = FilterConfig(retention_rates={
            domain: retention for domain in base_weights.keys()
        })

        mixture = DataMixture(weights=base_weights, filter_config=filter_config)
        curriculum = CurriculumSchedule.static(mixture)

        name = f"filter_{retention:.0%}".replace("%", "pct")
        trial_dir = bridge.create_trial_workspace(
            node_id=name,
            curriculum=curriculum,
            model_factory=args.model,
            max_steps=args.steps,
            batch_size=64 if "190M" in args.model else 256,
            sequence_length=4096,
        )

        experiments.append({
            "name": name,
            "retention_rate": retention,
            "workspace_dir": str(trial_dir),
            "model": args.model,
            "steps": args.steps,
        })
        logger.info(f"  Created: {name} (retention={retention:.0%}) → {trial_dir}")

    # Save experiment plan
    plan = {
        "experiment": "bitter_lesson_reasoning",
        "hypothesis": (
            "For reasoning tasks, the Bitter Lesson crossover (where less filtering "
            "beats more filtering) occurs later than for general LM, or does not occur "
            "at practical compute levels."
        ),
        "independent_variable": "filter_retention_rate",
        "dependent_variables": [
            "c4_val_loss (general LM)",
            "arc_easy (reasoning)",
            "gsm8k (math reasoning)",
            "piqa (commonsense)",
        ],
        "controlled": {
            "data_mixture": args.base_mix,
            "model": args.model,
            "steps": args.steps,
            "lr": 3e-4,
            "warmup": 500,
        },
        "experiments": experiments,
        "analysis_plan": [
            "1. Plot val_loss vs retention_rate → confirm basic Bitter Lesson",
            "2. Plot reasoning_accuracy vs retention_rate → test if crossover is delayed",
            "3. Compare crossover at 190M vs 3B → does model size shift the crossover?",
            "4. Decompose by domain: is math filtering more important than web filtering?",
        ],
    }

    plan_path = output_dir / "experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    logger.info(f"\nExperiment plan saved to {plan_path}")
    logger.info(f"{len(experiments)} trials ready to launch")

    # Generate launch script
    launch_script = output_dir / "launch_all.sh"
    lines = ["#!/bin/bash", "set -e", ""]
    for exp in experiments:
        workspace = exp["workspace_dir"]
        lines.append(f"echo 'Running {exp['name']}...'")
        lines.append(
            f"torchrun --nproc-per-node=8 "
            f"{Path(__file__).parent.parent / 'scripts' / 'train_olmo_3b.py'} "
            f"  # TODO: wire to workspace {workspace}"
        )
        lines.append("")
    launch_script.write_text("\n".join(lines))
    logger.info(f"Launch script: {launch_script}")


if __name__ == "__main__":
    main()
