#!/usr/bin/env python3
"""Test LLM-guided AutoML with 5 cycles on spaceship-titanic.

Quick test to validate:
1. LLM mutation proposer works correctly
2. Can read parent config and graph history
3. Proposes valid mutations
4. Integrates with MCGS framework

Strategy:
- Cycle 1-2: Rule-based model exploration (XGBoost, LightGBM)
- Cycle 3-5: LLM-guided hyperparameter tuning
"""

import os
import sys
from pathlib import Path

# Disable wandb
os.environ.setdefault("WANDB_DISABLED", "true")

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from agent_evolve.model.api import TrainingEvolver
from agent_evolve.model.types import TrainingEvolveConfig
from agent_evolve.model.algorithms.mcgs.search import MCGSSearch
from agent_evolve.model.algorithms.mcgs.llm_mutation import (
    LLMHyperparameterProposer,
)
from agent_evolve.model.algorithms.mcgs.ml_mutation import (
    MLModelTypeMutationProposer,
    CombinedMutationProposer,
)


class RootFanoutSelector:
    """Forces first N cycles to pick root as parent."""

    def __init__(self, fanout: int = 5) -> None:
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
    print("=" * 70)
    print("=== LLM-Guided AutoML Test (5 cycles) ===")
    print("=" * 70)
    print()
    print("Testing LLM mutation with Bedrock Claude Opus 4.6")
    print()
    print("Strategy:")
    print("  Cycle 1-2: Rule-based model exploration (XGBoost, LightGBM)")
    print("  Cycle 3-5: LLM-guided hyperparameter tuning")
    print()

    # Phase 1: Rule-based model exploration (2 cycles)
    phase1_mutator = MLModelTypeMutationProposer(
        model_types=("xgboost", "lightgbm")
    )

    # Phase 2: LLM-guided hyperparameter tuning (3 cycles)
    phase2_mutator = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
    )

    # Combine into phased strategy
    mutator = CombinedMutationProposer([
        phase1_mutator,  # cycle 1: XGBoost
        phase1_mutator,  # cycle 2: LightGBM
        phase2_mutator,  # cycle 3: LLM tuning
        phase2_mutator,  # cycle 4: LLM tuning
        phase2_mutator,  # cycle 5: LLM tuning
    ])

    # Use RootFanoutSelector to explore all 5 configurations
    selector = RootFanoutSelector(fanout=5)

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
            max_cycles=5,
            trial_budget_seconds=600,  # 10 min per trial
        ),
        work_dir=project_root / "runs" / "mle-automl-llm-test-5cycles",
    )

    print("Starting 5-cycle test run...")
    print()

    # Run evolution
    result = evolver.run(cycles=5)

    # Print results
    print("\n" + "=" * 70)
    print("=== Final Result ===")
    print("=" * 70)
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric: {result.best_metric}")
    print(f"graph_path: {result.graph_path}")
    print(f"report_path: {result.report_path}")

    if result.topk:
        print("\nTop 5 configurations:")
        for i, entry in enumerate(result.topk[:5], 1):
            print(
                f"  {i}. node={entry.node_id} "
                f"metric={entry.metric:.5f} "
                f"reward={entry.reward:.5f}"
            )

    # Print mutation history
    print("\n" + "=" * 70)
    print("=== Mutation History ===")
    print("=" * 70)

    # Read graph to see mutations
    import json
    with open(result.graph_path) as f:
        graph_data = json.load(f)

    for node in graph_data["nodes"]:
        if node["node_id"] == "node-root":
            continue
        print(f"\nNode: {node['node_id']}")
        print(f"  Mutation: {node['mutation_plan']}")
        print(f"  Metric: {node.get('metric', 'N/A')}")
        print(f"  Status: {node.get('trial_status', 'N/A')}")

    print("\n" + "=" * 70)
    print("Test complete! Check the mutation plans above to verify:")
    print("  - Cycles 1-2: Should show 'Switch to xgboost/lightgbm'")
    print("  - Cycles 3-5: Should show '[LLM] ...' with reasoning")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
