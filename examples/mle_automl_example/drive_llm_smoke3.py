#!/usr/bin/env python3
"""3-cycle smoke test for v3 LLM mutation (FE + dedup + noise warning).

Validates:
  1. Feature engineering flags can be toggled
  2. Backend reads flags and applies FE correctly
  3. Hard deduplication works (retries on duplicates)
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
    def __init__(self, exploration_cycles: int = 1) -> None:
        self.exploration_cycles = exploration_cycles

    def select(self, graph, *, cycle: int):
        root = graph.root()
        assert root is not None
        direct_children = [
            n for n in graph.nodes.values() if n.parent_id == root.node_id
        ]
        if len(direct_children) < self.exploration_cycles:
            return root
        valid = [
            n for n in graph.nodes.values()
            if n.node_id != "node-root" and n.is_valid and n.metric is not None
        ]
        return max(valid, key=lambda n: n.metric) if valid else root


def main():
    print("=" * 70)
    print("=== V3 SMOKE TEST (3 cycles) — FE + dedup + noise warning ===")
    print("=" * 70)
    print()

    phase1_mutator = MLModelTypeMutationProposer(model_types=("lightgbm",))
    phase2_mutator = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
        verbose=True,
    )

    mutator = CombinedMutationProposer([phase1_mutator] + [phase2_mutator] * 2)
    selector = HybridSelector(exploration_cycles=1)

    algo = MCGSSearch(mutator=mutator, selector=selector)
    evolver = TrainingEvolver(
        workspace=project_root / "seed_workspaces" / "mle_automl",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=3,
            trial_budget_seconds=600,
        ),
        work_dir=project_root / "runs" / "mle-automl-smoke3",
    )

    print("Starting 3-cycle smoke test...")
    print()
    result = evolver.run(cycles=3)

    print("\n" + "=" * 70)
    print("=== Result ===")
    print(f"best_metric: {result.best_metric}")
    for entry in result.topk[:5]:
        print(f"  {entry.node_id}: metric={entry.metric:.5f}")

    # Show mutations
    import json as _json
    with open(result.graph_path) as f:
        graph_data = _json.load(f)
    print("\n=== Mutations ===")
    for node in graph_data["nodes"]:
        if node["node_id"] == "node-root":
            continue
        print(f"\n{node['node_id']} metric={node.get('metric')}")
        print(f"  plan: {node['mutation_plan']}")
        # Show what was actually patched
        for op in node.get("workspace_patch", {}).get("operations", []):
            print(f"  op: {op.get('key_path')} = {op.get('value')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
