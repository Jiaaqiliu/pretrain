#!/usr/bin/env python3
"""Ensemble smoke test.

Cycles:
  1. LightGBM single baseline (rule)
  2. XGBoost single (rule)
  3. LightGBM single different seed (rule)
  4. Force ensemble via EnsembleMutationProposer — should combine top-3
  5. LLM proposes (sees ensemble in history, decides next step)

Validates:
  - EnsembleMutationProposer picks top-K and builds members
  - Backend trains members independently
  - EnsembleModel predicts correctly
  - metrics.json has ensemble_* secondary fields
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from agent_evolve.model.api import TrainingEvolver
from agent_evolve.model.types import (
    TrainingEvolveConfig,
    WorkspaceMutation,
    WorkspacePatch,
    PatchOperation,
)
from agent_evolve.model.algorithms.mcgs.search import MCGSSearch
from agent_evolve.model.algorithms.mcgs.llm_mutation import LLMHyperparameterProposer
from agent_evolve.model.algorithms.mcgs.ml_mutation import (
    MLModelTypeMutationProposer,
    CombinedMutationProposer,
    EnsembleMutationProposer,
)
import uuid


class HybridSelector:
    def __init__(self, exploration_cycles: int = 3):
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
    print("=== Ensemble Smoke Test (5 cycles) ===")
    print("=" * 70)

    # Cycles 1-3: explore base models
    model_sweep = MLModelTypeMutationProposer(
        model_types=("lightgbm", "xgboost", "random_forest")
    )
    # Cycle 4: bootstrap ensemble from top-3
    ensemble_rule = EnsembleMutationProposer(top_k=3, strategy="voting_soft")
    # Cycle 5: LLM
    llm = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
        verbose=True,
        workspace_root=project_root / "seed_workspaces" / "mle_automl",
    )

    mutator = CombinedMutationProposer([
        model_sweep, model_sweep, model_sweep,  # cycles 1-3
        ensemble_rule,                           # cycle 4
        llm,                                      # cycle 5
    ])
    selector = HybridSelector(exploration_cycles=3)

    algo = MCGSSearch(mutator=mutator, selector=selector)
    evolver = TrainingEvolver(
        workspace=project_root / "seed_workspaces" / "mle_automl",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(smoke=False, max_cycles=5, trial_budget_seconds=1200),
        work_dir=project_root / "runs" / "mle-automl-ensemble-smoke",
    )

    print("\nStarting ensemble smoke test...")
    result = evolver.run(cycles=5)

    print("\n" + "=" * 70)
    print(f"=== Result ===")
    print(f"best_metric (primary=Kaggle): {result.best_metric}")

    import json as _json
    with open(result.graph_path) as f:
        graph_data = _json.load(f)

    for node in graph_data["nodes"]:
        if node["node_id"] == "node-root":
            continue
        print(f"\n{node['node_id']}")
        print(f"  metric: {node.get('metric')}")
        print(f"  mutation: {node.get('mutation_plan', '')[:100]}")

        ckpt = node.get("checkpoint")
        if ckpt and ckpt.get("path"):
            from pathlib import Path as P
            mpath = P(ckpt["path"]).parents[2] / "evolution" / "eval" / "full_state" / "test" / "metrics.json"
            if mpath.exists():
                with open(mpath) as f:
                    m = _json.load(f)
                if "ensemble_strategy" in m:
                    print(f"  [ensemble] strategy={m['ensemble_strategy']} "
                          f"n_members={m['ensemble_n_members']} "
                          f"types={m.get('ensemble_member_types')}")
                if "cv_mean_accuracy" in m:
                    print(f"  [cv] mean={m['cv_mean_accuracy']:.5f} ± {m.get('cv_std', 0):.5f}")


if __name__ == "__main__":
    sys.exit(main())
