#!/usr/bin/env python3
"""Advanced 20-cycle AutoML search with feature engineering and diverse mutations.

This script demonstrates a more sophisticated search strategy:
1. Phase 1 (cycles 1-3): Explore model types
2. Phase 2 (cycles 4-10): Tune max_depth
3. Phase 3 (cycles 11-17): Tune n_estimators
4. Phase 4 (cycles 18-20): Random hyperparameter mutations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from agent_evolve.training.api import TrainingEvolver
from agent_evolve.training.algorithms.mcgs.ml_mutation import (
    MLModelTypeMutationProposer,
    MLDepthSweepProposer,
    MLNEstimatorsSweepProposer,
    MLLearningRateSweepProposer,
    MLHyperparameterMutationProposer,
    CombinedMutationProposer,
)
from agent_evolve.training.algorithms.mcgs.selectors import RootFanoutSelector


def main():
    print("=" * 60)
    print("=== Advanced 20-Cycle AutoML Search on MLE-Bench ===")
    print("=" * 60)
    print()
    print("Search Strategy:")
    print("  Phase 1 (3 cycles):  Model type exploration")
    print("  Phase 2 (7 cycles):  Depth tuning")
    print("  Phase 3 (7 cycles):  N-estimators tuning")
    print("  Phase 4 (3 cycles):  Random mutations")
    print()

    # Create evolver
    evolver = TrainingEvolver(
        workspace="seed_workspaces/mle_automl",
        run_name="mle-automl-advanced-20cycles",
        output_dir="runs",
    )

    # Configure MCGS
    evolver.config.max_cycles = 20
    evolver.config.trial_budget_seconds = 600.0  # 10 minutes per trial

    # Set backend and benchmark
    evolver.backend_name = "sklearn_backend"
    evolver.benchmark_name = "mle_bench"

    # Create combined mutator with phased strategy
    phase1_mutator = MLModelTypeMutationProposer(
        model_types=("xgboost", "lightgbm", "random_forest")
    )

    phase2_mutator = MLDepthSweepProposer(
        depths=(5, 8, 10, 12, 15, 20, 25)
    )

    phase3_mutator = MLNEstimatorsSweepProposer(
        n_estimators=(50, 100, 150, 200, 300, 400, 500)
    )

    phase4_mutator = MLHyperparameterMutationProposer(
        mutation_rate=0.3,
        random_state=42
    )

    # Combine into single mutator (will cycle through)
    mutator = CombinedMutationProposer([
        # Phase 1: 3 model types
        phase1_mutator,  # cycle 1
        phase1_mutator,  # cycle 2
        phase1_mutator,  # cycle 3

        # Phase 2: 7 depths
        phase2_mutator,  # cycles 4-10
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,
        phase2_mutator,

        # Phase 3: 7 n_estimators
        phase3_mutator,  # cycles 11-17
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,
        phase3_mutator,

        # Phase 4: 3 random mutations
        phase4_mutator,  # cycles 18-20
        phase4_mutator,
        phase4_mutator,
    ])

    # Use RootFanoutSelector to explore all 20 configurations
    selector = RootFanoutSelector(num_children=20)

    print("Starting 20-cycle search...")
    print()

    # Run evolution
    result = evolver.evolve(
        mutator=mutator,
        selector=selector,
    )

    # Print results
    print("\n" + "=" * 60)
    print("=== Final Result ===")
    print("=" * 60)
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric: {result.best_metric}")
    print(f"graph_path: {result.graph_path}")
    print(f"report_path: {result.report_path}")

    if result.topk:
        print("\nTop 10 configurations:")
        for i, entry in enumerate(result.topk[:10], 1):
            print(
                f"  {i:2d}. node={entry.node_id} branch={entry.branch_id} "
                f"metric={entry.metric:.4f} reward={entry.reward:.4f}"
            )

    print("\nBest configuration saved to:")
    incumbent_path = Path(result.graph_path).parent / 'incumbent' / 'model' / 'config.yaml'
    print(f"  {incumbent_path}")

    # Print improvement over baseline
    if result.best_metric and result.best_metric > 0:
        baseline = 0.77816  # Previous 4-cycle baseline
        improvement = result.best_metric - baseline
        improvement_pct = (improvement / baseline) * 100

        print(f"\nImprovement over baseline:")
        print(f"  Baseline (4-cycle):  {baseline:.5f}")
        print(f"  Advanced (20-cycle): {result.best_metric:.5f}")
        print(f"  Improvement:         +{improvement:.5f} ({improvement_pct:+.2f}%)")

        # Check if we reached top 20
        top_20_threshold = 0.82183
        if result.best_metric >= top_20_threshold:
            print(f"\n🎉 SUCCESS! Reached top 20 threshold ({top_20_threshold:.5f})")
        else:
            gap = top_20_threshold - result.best_metric
            print(f"\n📊 Gap to top 20: {gap:.5f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
