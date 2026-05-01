#!/usr/bin/env python3
"""20-cycle LLM-guided AutoML search on Spaceship Titanic.

Applies all 4 optimizations:
  1. Full config history in LLM context (not just descriptions)
  2. HybridSelector: rule exploration → LLM exploitation of topk
  3. Tried-configs tracking to avoid repetition
  4. Expanded search space (regularization, feature engineering)

Strategy:
  Phase 1 (cycles 1-3):   Rule-based model exploration (XGBoost, LightGBM, RandomForest)
  Phase 2 (cycles 4-20):  LLM-guided multi-parameter mutations based on topk

Expected: Should beat 20-cycle rule-based baseline (0.81839) by leveraging
intelligent exploration and multi-parameter coordination.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")

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


class HybridSelector:
    """Phase-aware parent selector.

    - Cycles 1..exploration_cycles: Always pick root (explore models)
    - Cycles exploration_cycles+1..∞: Pick best valid non-root node (exploit)

    This enables LLM to EVOLVE from the best-seen config instead of
    always starting from scratch.
    """

    def __init__(self, exploration_cycles: int = 3) -> None:
        self.exploration_cycles = exploration_cycles

    def select(self, graph, *, cycle: int):
        root = graph.root()
        assert root is not None

        # Count how many direct children of root we already have
        direct_children = [
            n for n in graph.nodes.values() if n.parent_id == root.node_id
        ]

        # Phase 1: ensure we have N explored siblings at root
        if len(direct_children) < self.exploration_cycles:
            return root

        # Phase 2: pick the best valid node (not necessarily a root child)
        valid = [
            n for n in graph.nodes.values()
            if n.node_id != "node-root"
            and n.is_valid
            and n.metric is not None
        ]
        if not valid:
            return root

        # Exploit the current best
        return max(valid, key=lambda n: n.metric)


def main():
    print("=" * 70)
    print("=== LLM-Guided 20-Cycle AutoML Search (Opus 4.7) ===")
    print("=" * 70)
    print()
    print("Optimizations applied:")
    print("  #1 Full config history in LLM context")
    print("  #2 HybridSelector: rule exploration → LLM exploitation")
    print("  #3 Tried-configs tracking (avoid repetition)")
    print("  #4 Expanded search space (regularization + features)")
    print()
    print("Strategy:")
    print("  Phase 1 (cycles 1-3):   Rule-based model exploration")
    print("  Phase 2 (cycles 4-20):  LLM-guided evolution from topk")
    print()
    print("Baseline to beat: 0.81839 (20-cycle rule-based)")
    print()

    # Phase 1: Rule-based model exploration (3 cycles)
    phase1_mutator = MLModelTypeMutationProposer(
        model_types=("xgboost", "lightgbm", "random_forest")
    )

    # Phase 2: LLM-guided evolution (17 cycles)
    phase2_mutator = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
        verbose=True,
    )

    # Combine: 3 rule + 17 LLM
    mutators = [phase1_mutator] * 3 + [phase2_mutator] * 17
    mutator = CombinedMutationProposer(mutators)

    # HybridSelector: first 3 cycles from root, then exploit topk
    selector = HybridSelector(exploration_cycles=3)

    # MCGS
    algo = MCGSSearch(mutator=mutator, selector=selector)

    # Evolver
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
        work_dir=project_root / "runs" / "mle-automl-llm-20cycles",
    )

    print("Starting 20-cycle LLM-guided search...")
    print()

    result = evolver.run(cycles=20)

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
        print("\nTop 10 configurations:")
        for i, entry in enumerate(result.topk[:10], 1):
            print(
                f"  {i:2d}. node={entry.node_id} "
                f"metric={entry.metric:.5f} "
                f"reward={entry.reward:.5f}"
            )

    # Print full mutation history
    print("\n" + "=" * 70)
    print("=== Mutation History ===")
    print("=" * 70)

    import json as _json
    with open(result.graph_path) as f:
        graph_data = _json.load(f)

    for node in graph_data["nodes"]:
        if node["node_id"] == "node-root":
            continue
        metric = node.get("metric", "N/A")
        metric_str = f"{metric:.5f}" if isinstance(metric, (int, float)) else str(metric)
        marker = " ⭐" if metric == result.best_metric else ""
        print(f"\nNode: {node['node_id']}{marker}")
        print(f"  Parent: {node.get('parent_id', 'N/A')}")
        print(f"  Metric: {metric_str}")
        print(f"  Mutation: {node['mutation_plan'][:200]}")

    # Compare with baseline
    if result.best_metric and result.best_metric > 0:
        baseline = 0.81839  # 20-cycle rule-based
        improvement = result.best_metric - baseline
        improvement_pct = (improvement / baseline) * 100

        print(f"\n" + "=" * 70)
        print("=== Comparison with Rule-based Baseline ===")
        print("=" * 70)
        print(f"  Rule-based (20-cycle):  {baseline:.5f}")
        print(f"  LLM-guided (20-cycle):  {result.best_metric:.5f}")
        print(f"  Improvement:            {improvement:+.5f} ({improvement_pct:+.2f}%)")

        if improvement > 0:
            print(f"\n✨ LLM-guided search IMPROVED by {improvement:.5f}!")
        elif improvement == 0:
            print(f"\n📊 LLM-guided search matched baseline")
        else:
            print(f"\n❌ LLM-guided search below baseline (may need more cycles)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
