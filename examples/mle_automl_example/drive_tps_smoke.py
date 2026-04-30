#!/usr/bin/env python3
"""Smoke test on TPS May 2022 — 3 cycles to validate:
  1. Generic BaseTabularFeatureEngineer works (features are all f_00..f_30, anonymous).
  2. Backend + benchmark grade via mlebench AUC grader.
  3. LLM proposer works on a dataset it has no prior for.
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
)


class HybridSelector:
    def __init__(self, exploration_cycles: int = 1):
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
    print("=== TPS May 2022 Smoke Test (3 cycles) ===")
    print("=" * 70)
    print("Validates: Generic FE + AUC grader + LLM on new dataset")
    print()

    phase1 = MLModelTypeMutationProposer(model_types=("xgboost", "lightgbm"))
    llm = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
        verbose=True,
        workspace_root=project_root / "seed_workspaces" / "mle_automl_tps",
    )

    mutator = CombinedMutationProposer([phase1, phase1, llm])
    selector = HybridSelector(exploration_cycles=2)
    algo = MCGSSearch(mutator=mutator, selector=selector)

    evolver = TrainingEvolver(
        workspace=project_root / "seed_workspaces" / "mle_automl_tps",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=3,
            trial_budget_seconds=1200,
        ),
        work_dir=project_root / "runs" / "mle-automl-tps-smoke",
    )

    print("Starting 3-cycle smoke test...")
    result = evolver.run(cycles=3)

    print("\n" + "=" * 70)
    print("=== Result ===")
    print(f"best_metric (AUC): {result.best_metric}")

    import json as _json
    with open(result.graph_path) as f:
        graph_data = _json.load(f)
    for node in graph_data["nodes"]:
        if node["node_id"] == "node-root":
            continue
        print(f"\n{node['node_id']}")
        print(f"  metric: {node.get('metric')}")
        print(f"  mutation: {node.get('mutation_plan', '')[:120]}")


if __name__ == "__main__":
    sys.exit(main())
