#!/usr/bin/env python3
"""20-cycle v5: ensembles enabled.

Schedule:
  Cycles 1-3:   Rule — explore base models (xgboost, lightgbm, random_forest)
  Cycles 4-15:  LLM-guided — single-model mutations (FE flags, hyperparameters)
  Cycles 16-17: Rule — EnsembleMutationProposer bootstraps ensembles from current top-K
  Cycles 18-20: LLM — refine ensembles (change members, strategy, etc.)

Compared to v4 (single-model only, best 0.83218 on spaceship-titanic),
v5 gives LLM access to ensemble layer + has a rule-based ensemble
bootstrapper so LLM doesn't have to reinvent member composition from scratch.
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
    print("=== LLM-Guided 20-Cycle AutoML v5 (ensembles + CV + FE) ===")
    print("=" * 70)
    print()
    print("Schedule:")
    print("  Cycles  1-3:  Rule — base-model exploration")
    print("  Cycles  4-15: LLM — single-model FE/hyperparameter mutations")
    print("  Cycles 16-17: Rule — EnsembleMutationProposer (top-K → ensemble)")
    print("  Cycles 18-20: LLM — refine ensemble composition")
    print()
    print("Baseline to beat: 0.83218 (LightGBM defaults, v4)")
    print()

    model_sweep = MLModelTypeMutationProposer(
        model_types=("xgboost", "lightgbm", "random_forest")
    )
    llm = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
        verbose=True,
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
        workspace=project_root / "seed_workspaces" / "mle_automl",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=20,
            trial_budget_seconds=1800,
        ),
        work_dir=project_root / "runs" / "mle-automl-llm-20cycles-v5",
    )

    print("Starting 20-cycle v5 search...")
    result = evolver.run(cycles=20)

    print("\n" + "=" * 70)
    print("=== Final Result ===")
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric (Kaggle primary): {result.best_metric}")

    if result.topk:
        print("\nTop 10:")
        for i, entry in enumerate(result.topk[:10], 1):
            print(f"  {i:2d}. node={entry.node_id} metric={entry.metric:.5f}")

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
        print(f"  Parent: {node.get('parent_id')}")
        print(f"  Metric: {metric_str}")
        print(f"  Mutation: {node['mutation_plan'][:200]}")

        # Secondary metrics
        ckpt = node.get("checkpoint")
        if ckpt and ckpt.get("path"):
            from pathlib import Path as P
            mpath = P(ckpt["path"]).parents[2] / "evolution" / "eval" / "full_state" / "test" / "metrics.json"
            if mpath.exists():
                with open(mpath) as f:
                    m = _json.load(f)
                secondaries = []
                if "ensemble_strategy" in m:
                    secondaries.append(
                        f"ensemble={m['ensemble_strategy']}/{m['ensemble_n_members']}"
                    )
                if "cv_mean_accuracy" in m:
                    secondaries.append(
                        f"cv={m['cv_mean_accuracy']:.4f}±{m.get('cv_std', 0):.4f}"
                    )
                if secondaries:
                    print(f"  Secondary: {' | '.join(secondaries)}")

    baseline = 0.83218
    if result.best_metric and result.best_metric > 0:
        improvement = result.best_metric - baseline
        pct = improvement / baseline * 100
        print(f"\n{'=' * 70}")
        print(f"vs v4 baseline (0.83218): {improvement:+.5f} ({pct:+.2f}%)")
        if improvement > 0:
            print(f"✨ IMPROVED by {improvement:+.5f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
