#!/usr/bin/env python3
"""20-cycle LLM AutoML on TPS May 2022.

Same schedule as drive_llm_20cycle_v5.py but pointing at the TPS workspace
(subsampled to 100K rows for tractable training). Validates framework
generalization: zero code changes from spaceship-titanic.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from agent_evolve.training.api import TrainingEvolver
from agent_evolve.training.types import TrainingEvolveConfig
from agent_evolve.training.algorithms.mcgs.search import MCGSSearch
from agent_evolve.training.algorithms.mcgs.llm_mutation import LLMHyperparameterProposer
from agent_evolve.training.algorithms.mcgs.ml_mutation import (
    MLModelTypeMutationProposer,
    CombinedMutationProposer,
    EnsembleMutationProposer,
)


class HybridSelector:
    def __init__(self, exploration_cycles: int = 3) -> None:
        self.exploration_cycles = exploration_cycles

    def select(self, graph, *, cycle: int):
        root = graph.root()
        direct = [n for n in graph.nodes.values() if n.parent_id == root.node_id]
        if len(direct) < self.exploration_cycles:
            return root
        valid = [n for n in graph.nodes.values()
                 if n.node_id != "node-root" and n.is_valid and n.metric is not None]
        return max(valid, key=lambda n: n.metric) if valid else root


def main():
    print("=" * 70)
    print("=== TPS May 2022 — 20-cycle AutoML ===")
    print("=" * 70)

    model_sweep = MLModelTypeMutationProposer(
        model_types=("xgboost", "lightgbm", "random_forest")
    )
    llm = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
        verbose=True,
        workspace_root=project_root / "seed_workspaces" / "mle_automl_tps",
    )
    ensemble_rule = EnsembleMutationProposer(top_k=3, strategy="voting_soft")

    mutators = (
        [model_sweep] * 3
        + [llm] * 12
        + [ensemble_rule] * 2
        + [llm] * 3
    )
    assert len(mutators) == 20
    mutator = CombinedMutationProposer(mutators)

    selector = HybridSelector(exploration_cycles=3)
    algo = MCGSSearch(mutator=mutator, selector=selector)

    evolver = TrainingEvolver(
        workspace=project_root / "seed_workspaces" / "mle_automl_tps",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=20,
            trial_budget_seconds=1800,
        ),
        work_dir=project_root / "runs" / "mle-automl-tps-20cycles",
    )

    print("Starting 20-cycle TPS run...")
    result = evolver.run(cycles=20)

    print("\n" + "=" * 70)
    print("=== Final Result ===")
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"best_metric (AUC primary): {result.best_metric}")

    if result.topk:
        print("\nTop 10:")
        for i, e in enumerate(result.topk[:10], 1):
            print(f"  {i:2d}. {e.node_id}: {e.metric:.5f}")

    print("\n" + "=" * 70)
    print("=== Mutation History ===")
    print("=" * 70)

    import json as _json
    with open(result.graph_path) as f:
        graph_data = _json.load(f)

    for node in graph_data["nodes"]:
        if node["node_id"] == "node-root":
            continue
        metric = node.get("metric")
        metric_str = f"{metric:.5f}" if isinstance(metric, (int, float)) else str(metric)
        marker = " ⭐" if metric == result.best_metric else ""
        print(f"\nNode: {node['node_id']}{marker}")
        print(f"  Metric: {metric_str}")
        print(f"  Mutation: {node['mutation_plan'][:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
