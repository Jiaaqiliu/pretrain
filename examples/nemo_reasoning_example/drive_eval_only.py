"""Eval-only cycle: load an existing LoRA adapter and score it on Kaggle dev.

Runs TrainingEvolver with ``cycles=1``, a single mutation that does nothing
(just picks up the seed workspace as-is), and relies on the seed's
``model/adapter.yaml::seed_adapter_path`` to skip training and drop straight
into vLLM + LoRA eval on the 951-row dev split.

Use this to:
  * Reproduce E-28's 49.63% dev (set seed_adapter_path to E-28's adapter).
  * Spot-check an adapter your own training cycle produced earlier.
  * Validate that your dev CSV + model path + tokenizer are wired end-to-end
    before burning GPU hours on real SFT.

Launch: see ``run_eval_only.sh``.
Wallclock: ~10 min on 1× H200.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

AE = Path("/fsx/zzsamshi/a-evolve")
sys.path.insert(0, str(AE))

from agent_evolve.training.algorithms.mcgs.search import MCGSSearch  # noqa: E402
from agent_evolve.training.api import TrainingEvolver  # noqa: E402
from agent_evolve.training.types import (  # noqa: E402
    PatchOperation,
    TrainingEvolveConfig,
    WorkspaceMutation,
    WorkspacePatch,
)


SEED_ADAPTER = "/fsx/zzsamshi/nemotron-auto-research/experiments/E-28-iter3-noprm/adapter"


class NoOpMutator:
    """Minimal mutator — injects the seed adapter path into the candidate
    workspace and adds a harmless annotation so the fork still differs from
    the seed (required by validate_training_workspace).

    Patching ``model/adapter.yaml`` via mutation rather than editing the seed
    workspace on disk keeps eval-only runs from colliding with SFT/GSPO
    drivers, which need the seed adapter unset.
    """

    def propose(self, parent, graph=None):  # noqa: ARG002
        return WorkspaceMutation(
            mutation_id="m-eval-only",
            parent_node_id=parent.node_id,
            description="eval-only (no training)",
            patch=WorkspacePatch(operations=[
                PatchOperation(
                    op="replace",
                    path="model/adapter.yaml",
                    key_path=["seed_adapter_path"],
                    value=SEED_ADAPTER,
                ),
                PatchOperation(
                    op="replace",
                    path="model/adapter.yaml",
                    key_path=["seed_adapter_name"],
                    value="E-28-iter3-noprm",
                ),
                PatchOperation(
                    op="replace",
                    path="data/curriculum.yaml",
                    key_path=["annotation"],
                    value="eval-only-cycle",
                ),
            ]),
            mutation_type="debug",
        )


def main() -> int:
    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm=MCGSSearch(mutator=NoOpMutator()),
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=1,
            trial_budget_seconds=1800,  # 30 min hard cap (~10 min expected)
        ),
        work_dir=AE / "runs" / "eval-only",
    )
    result = evolver.run(cycles=1)
    print("\n=== Final Result ===")
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric: {result.best_metric}")
    print(f"graph_path: {result.graph_path}")
    print(f"report_path: {result.report_path}")
    for entry in result.topk:
        print(
            f"  topk: node={entry.node_id} branch={entry.branch_id} "
            f"metric={entry.metric} reward={entry.reward}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
