#!/usr/bin/env python3
"""Advanced 20-cycle AutoML search with feature engineering and diverse mutations.

This script demonstrates a more sophisticated search strategy:
1. Phase 1 (cycles 1-3): Explore model types
2. Phase 2 (cycles 4-10): Tune max_depth
3. Phase 3 (cycles 11-17): Tune n_estimators
4. Phase 4 (cycles 18-20): Random hyperparameter mutations
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
from agent_evolve.training.algorithms.mcgs.ml_mutation import (
    MLModelTypeMutationProposer,
    MLDepthSweepProposer,
    MLNEstimatorsSweepProposer,
    MLLearningRateSweepProposer,
    MLHyperparameterMutationProposer,
    CombinedMutationProposer,
)


class RootFanoutSelector:
    """Forces first `fanout` cycles to pick root as parent.

    Ensures we get independent siblings testing different models,
    rather than a chain from UCT.
    """

    def __init__(self, fanout: int = 4) -> None:
        self.fanout = fanout

    def select(self, graph, *, cycle: int):
        root = graph.root()
        assert root is not None
        direct_children = [
            n for n in graph.nodes.values() if n.parent_id == root.node_id
        ]
        if len(direct_children) < self.fanout:
            return root
        # Fallback: pick best child
        return max(
            direct_children,
            key=lambda n: (n.mean_reward, n.metric or float("-inf")),
        )


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
    selector = RootFanoutSelector(fanout=20)

    # Create MCGS algorithm
    algo = MCGSSearch(
        mutator=mutator,
        selector=selector,
    )

    # Create evolver
    evolver = TrainingEvolver(
        workspace=project_root / "seed_workspaces" / "mle_automl",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=20,
            trial_budget_seconds=600,  # 10 min per trial
        ),
        work_dir=project_root / "runs" / "mle-automl-advanced-20cycles",
    )

    print("Starting 20-cycle search...")
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
