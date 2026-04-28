#!/usr/bin/env python3
"""CV-specific smoke test: force CV on cycle 1 and verify it works correctly.

Validates:
  1. Backend reads eval/cv.yaml with enabled=true
  2. Training produces K fold models + CV-mean + CV-std
  3. metrics.json has Kaggle score as PRIMARY, CV metrics as SECONDARY
  4. LLM context for subsequent cycles includes the CV secondary
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from agent_evolve.training.api import TrainingEvolver
from agent_evolve.training.types import (
    TrainingEvolveConfig,
    WorkspaceMutation,
    WorkspacePatch,
    PatchOperation,
)
from agent_evolve.training.algorithms.mcgs.search import MCGSSearch
from agent_evolve.training.algorithms.mcgs.llm_mutation import LLMHyperparameterProposer
import uuid


class ForcedCVMutator:
    """Cycle 1: Switch to lightgbm + enable 5-fold CV (forced).
    Cycle 2+: Delegate to LLM."""

    def __init__(self, llm_proposer):
        self._i = 0
        self.llm = llm_proposer

    def propose(self, parent, graph):
        self._i += 1
        if self._i == 1:
            # Force: lightgbm defaults + CV on
            return WorkspaceMutation(
                mutation_id=f"m-forced-{uuid.uuid4().hex[:8]}",
                parent_node_id=parent.node_id,
                description="FORCED: LightGBM defaults + 5-fold CV",
                patch=WorkspacePatch(operations=[
                    PatchOperation(
                        op="replace",
                        path="model/config.yaml",
                        key_path=["model_type"],
                        value="lightgbm",
                    ),
                    PatchOperation(
                        op="replace",
                        path="eval/cv.yaml",
                        key_path=["enabled"],
                        value=True,
                    ),
                    PatchOperation(
                        op="replace",
                        path="eval/cv.yaml",
                        key_path=["n_splits"],
                        value=5,
                    ),
                ]),
                mutation_type="training_recipe",
            )
        # Cycle 2+ delegate to LLM to see if it picks up the secondary metric
        return self.llm.propose(parent, graph)


class HybridSelector:
    def __init__(self, exploration_cycles: int = 1):
        self.exploration_cycles = exploration_cycles

    def select(self, graph, *, cycle: int):
        root = graph.root()
        direct_children = [n for n in graph.nodes.values() if n.parent_id == root.node_id]
        if len(direct_children) < self.exploration_cycles:
            return root
        valid = [n for n in graph.nodes.values()
                 if n.node_id != "node-root" and n.is_valid and n.metric is not None]
        return max(valid, key=lambda n: n.metric) if valid else root


def main():
    print("=" * 70)
    print("=== CV Smoke Test ===")
    print("=" * 70)
    print("Cycle 1: forced LightGBM + 5-fold CV")
    print("Cycle 2: LLM sees CV in secondary metrics, decides next step")
    print()

    llm = LLMHyperparameterProposer(
        model_id="us.anthropic.claude-opus-4-7",
        region="us-west-2",
        verbose=True,
    )
    mutator = ForcedCVMutator(llm_proposer=llm)
    selector = HybridSelector(exploration_cycles=1)

    algo = MCGSSearch(mutator=mutator, selector=selector)
    evolver = TrainingEvolver(
        workspace=project_root / "seed_workspaces" / "mle_automl",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(smoke=False, max_cycles=2, trial_budget_seconds=600),
        work_dir=project_root / "runs" / "mle-automl-cv-smoke",
    )

    result = evolver.run(cycles=2)

    print("\n=== Result ===")
    print(f"best_metric (primary = Kaggle holdout): {result.best_metric}")

    # Inspect metrics.json of cycle 1
    import json as _json
    with open(result.graph_path) as f:
        graph_data = _json.load(f)

    print("\n=== Per-node summary ===")
    for node in graph_data["nodes"]:
        if node["node_id"] == "node-root":
            continue
        nid = node["node_id"]
        print(f"\n{nid}")
        print(f"  primary metric (Kaggle): {node.get('metric')}")
        print(f"  mutation: {node['mutation_plan']}")

        # Try load metrics.json
        ckpt = node.get("checkpoint", {})
        if ckpt and ckpt.get("path"):
            from pathlib import Path as P
            metrics_path = P(ckpt["path"]).parents[2] / "evolution" / "eval" / "full_state" / "test" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = _json.load(f)
                print(f"  metrics.json primary_metric: {m.get('primary_metric')}")
                print(f"  metrics.json mle_bench_score: {m.get('mle_bench_score')}")
                if "cv_mean_accuracy" in m:
                    print(f"  metrics.json cv_mean_accuracy: {m['cv_mean_accuracy']:.5f}")
                    print(f"  metrics.json cv_std: {m.get('cv_std', 'N/A')}")
                    print(f"  metrics.json cv_n_splits: {m.get('cv_n_splits')}")


if __name__ == "__main__":
    sys.exit(main())
