"""1-cycle GSPO-only run, MCGS-driven.

Single cycle, single parent (root), single child. The mutator flips the
``rl_gspo`` stage in ``train/pipeline.yaml`` from ``enabled: false`` to
``enabled: true`` AND disables the ``sft_warmup`` stage, so the run does:

  1. Rollout G=n_samples completions per prompt using the seed LoRA adapter
     (``model/adapter.yaml::seed_adapter_path``) via vLLM + LoRA.
  2. Group-normalize advantages within (domain, pid).
  3. GSPO / DAPO sequence-level clipped update on the adapter.
  4. Eval the updated adapter on Kaggle dev.

Uses the TinkerLite ``SamplingClient`` + ``TrainingClient`` protocols end-to-end
(``agent_evolve.backends.tinkerlite.hf_clients.HFTrainingClient`` +
``VLLMSamplingClient``). Mirrors the verified recipe in
``../nemotron-auto-research/scripts/gspo_rollout.py`` + ``scripts/gspo_update.py``.

Launch: see ``run_gspo_1cycle.sh``.
Budget: ~60-90 min (rollout ~15-25 min at G=4 × 50 prompts, update ~20-30 min).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP4", "0")
os.environ.setdefault("VLLM_ALLREDUCE_USE_FLASHINFER", "0")

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


class EnableGSPOMutator:
    """One-shot mutator that swaps SFT → GSPO in ``train/pipeline.yaml``.

    - Injects ``model/adapter.yaml::seed_adapter_path`` so the training
      client + rollout sampling client both start from E-28 (the SFT
      baseline) rather than from the bare base model.
    - Disables the ``sft_warmup`` stage (index 1) so this cycle is RL-only.
    - Enables the ``rl_gspo`` stage (index 2).
    - Flags the seed adapter as override-eligible so ``_run_pipeline`` runs
      the RL stage instead of short-circuiting on a seed adapter passthrough.
    """

    def propose(self, parent, graph=None):  # noqa: ARG002
        return WorkspaceMutation(
            mutation_id="m-enable-gspo",
            parent_node_id=parent.node_id,
            description="Enable rl_gspo stage, disable sft_warmup",
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
                    path="train/pipeline.yaml",
                    key_path=["override_seed_adapter"],
                    value=True,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 1, "enabled"],
                    value=False,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "enabled"],
                    value=True,
                ),
            ]),
            mutation_type="training_recipe",
        )


def main() -> int:
    algo = MCGSSearch(mutator=EnableGSPOMutator())
    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm=algo,
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=1,
            # Budget must cover: vLLM load (~2m) + rollout (~15-25m) +
            # update (~20-30m) + vLLM eval load+gen (~10m).
            trial_budget_seconds=5400,  # 90 min hard cap
        ),
        work_dir=AE / "runs" / "gspo-1cycle",
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
