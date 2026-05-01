"""DDP shakeout: 1 GSPO cycle with max_steps=4 to verify torchrun-based
DDP training fires + all 8 GPUs participate.

Launch with AE_TRAIN_DDP=1 and CUDA_VISIBLE_DEVICES unset.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP4", "0")
os.environ.setdefault("VLLM_ALLREDUCE_USE_FLASHINFER", "0")

AE = Path("/fsx/zzsamshi/a-evolve")
NAR = Path("/fsx/zzsamshi/nemotron-auto-research")
sys.path.insert(0, str(AE))

from agent_evolve.model.algorithms.mcgs.search import MCGSSearch  # noqa: E402
from agent_evolve.model.api import TrainingEvolver  # noqa: E402
from agent_evolve.model.types import (  # noqa: E402
    PatchOperation,
    TrainingEvolveConfig,
    WorkspaceMutation,
    WorkspacePatch,
)

SEED_E28 = str(NAR / "experiments" / "E-28-iter3-noprm" / "adapter")


class DDPShakeMutator:
    """Tiny GSPO cycle: small rollout pool + 4 opt steps. Enough to verify
    the DDP update path fires and all 8 GPUs actively train."""

    def propose(self, parent, graph=None):  # noqa: ARG002
        return WorkspaceMutation(
            mutation_id=f"m-ddp-shake-{uuid.uuid4().hex[:6]}",
            parent_node_id=parent.node_id,
            description="DDP shakeout: per_domain=10 G=4 max_steps=4",
            patch=WorkspacePatch(operations=[
                PatchOperation(
                    op="replace",
                    path="model/adapter.yaml",
                    key_path=["seed_adapter_path"],
                    value=SEED_E28,
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
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "per_domain"],
                    value=10,   # 60 prompts total → 240 rollouts @ G=4
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "n_samples"],
                    value=4,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "max_tokens"],
                    value=1024,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "max_len"],
                    value=1400,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "grad_accum"],
                    value=4,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "max_steps"],
                    value=4,    # 4 opt steps × grad_accum=4 = 16 micro-steps per rank
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "log_every"],
                    value=1,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "lr"],
                    value=3.0e-6,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "eps_low"],
                    value=3.0e-4,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "eps_high"],
                    value=4.0e-4,
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "advantage_mode"],
                    value="group",
                ),
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 2, "domains"],
                    value=["bits", "cipher", "equations", "gravity", "units", "numerals"],
                ),
            ]),
            mutation_type="training_recipe",
        )


def main() -> int:
    algo = MCGSSearch(mutator=DDPShakeMutator())
    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm=algo,
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=1,
            trial_budget_seconds=3600,  # 1hr hard cap
        ),
        work_dir=AE / "runs" / "ddp-shakeout",
    )
    result = evolver.run(cycles=1)
    print("\n=== DDP Shakeout Result ===")
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric: {result.best_metric}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
