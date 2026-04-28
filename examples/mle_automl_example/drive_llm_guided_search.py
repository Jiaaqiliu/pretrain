#!/usr/bin/env python3
"""LLM-guided AutoML search - intelligent mutation based on training history.

Instead of predefined sweeps, this uses Claude to reason about:
- What worked and what didn't in previous trials
- Overfitting/underfitting signals
- Feature engineering opportunities
- Smart hyperparameter adjustments

The LLM analyzes the MCGS graph and proposes context-aware mutations.
"""

import os
import sys
from pathlib import Path

# Disable wandb
os.environ.setdefault("WANDB_DISABLED", "true")

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from agent_evolve.training.api import TrainingEvolver
from agent_evolve.training.types import TrainingEvolveConfig
from agent_evolve.training.algorithms.mcgs.search import MCGSSearch
from agent_evolve.training.algorithms.mcgs.llm_mutation import (
    LLMFeatureEngineeringProposer,
    LLMHyperparameterProposer,
)
from agent_evolve.training.algorithms.mcgs.ml_mutation import (
    MLModelTypeMutationProposer,
    CombinedMutationProposer,
)


def main():
    print("=" * 60)
    print("=== LLM-Guided AutoML Search ===")
    print("=" * 60)
    print()
    print("Strategy:")
    print("  Phase 1 (3 cycles):  Rule-based model type exploration")
    print("  Phase 2 (7 cycles):  LLM-guided hyperparameter tuning")
    print("  Phase 3 (10 cycles): LLM-guided feature engineering")
    print()

    # Phase 1: Start with rule-based model exploration
    phase1_mutator = MLModelTypeMutationProposer(
        model_types=("xgboost", "lightgbm", "random_forest")
    )

    # Phase 2: LLM analyzes results and tunes hyperparameters
    phase2_mutator = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
    )

    # Phase 3: LLM proposes feature engineering strategies
    phase3_mutator = LLMFeatureEngineeringProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
    )

    # Combine into phased strategy
    mutator = CombinedMutationProposer([
        # Phase 1: 3 model types
        phase1_mutator,
        phase1_mutator,
        phase1_mutator,

        # Phase 2: 7 LLM-guided hyperparameter tuning
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,

        # Phase 3: 10 LLM-guided feature engineering
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
    ])

    # Create MCGS algorithm
    algo = MCGSSearch(mutator=mutator)

    # Create evolver
    evolver = TrainingEvolver(
        workspace=project_root / "seed_workspaces" / "mle_automl",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=20,
            trial_budget_seconds=600,
        ),
        work_dir=project_root / "runs" / "mle-automl-llm-guided",
    )

    print("Starting LLM-guided 20-cycle search...")
    print("Note: LLM will analyze training history and propose intelligent mutations")
    print()

    # Run evolution
    result = evolver.run(cycles=20)

    # Print results
    print("\n" + "=" * 60)
    print("=== Final Result ===")
    print("=" * 60)
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric: {result.best_metric}")
    print(f"graph_path: {result.graph_path}")

    if result.topk:
        print("\nTop 10 configurations:")
        for i, entry in enumerate(result.topk[:10], 1):
            print(
                f"  {i:2d}. node={entry.node_id} "
                f"metric={entry.metric:.4f} "
                f"mutation={entry.branch_id}"
            )

    # Compare with rule-based baseline
    if result.best_metric and result.best_metric > 0:
        baseline = 0.81839  # Previous rule-based 20-cycle
        improvement = result.best_metric - baseline

        print(f"\nComparison:")
        print(f"  Rule-based (20-cycle):  {baseline:.5f}")
        print(f"  LLM-guided (20-cycle):  {result.best_metric:.5f}")
        print(f"  Delta:                  {improvement:+.5f}")

        if improvement > 0:
            print(f"\n✨ LLM-guided search improved by {improvement:.5f}!")
        else:
            print(f"\n📊 LLM-guided search was comparable to rule-based")

    return 0


if __name__ == "__main__":
    sys.exit(main())
