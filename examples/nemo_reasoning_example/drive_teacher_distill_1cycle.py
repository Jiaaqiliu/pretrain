"""1-cycle teacher-distillation → SFT → eval run, MCGS-driven.

Single cycle, single parent (root), single child:

  1. ``synth_generate`` stage uses Nemotron-Super-120B-FP8 (TP=4) to sample
     500 prompts (250 cipher + 250 bits) from the Kaggle train CSV, filters
     by verifier-correct + has_boxed + min_tokens≥2500 + student_len≤8192.
     The kept JSONL is appended to ``data/sources.yaml``.
  2. ``sft_warmup`` stage trains rank-16 LoRA on the mix of ``short_correct.jsonl``
     plus the teacher traces.
  3. Eval on the 951-row Kaggle dev via vLLM + LoRA.
  4. MCGS sees one score, sets it as incumbent.

Per CLAUDE.md §E-38: min_tokens=2500 (recovered from the failed 5000 gate),
max_tokens=8192 (bits still sometimes hits this). Budget: ~75-95 min total
on 4×H200 for synth + 1×H200 for SFT/eval.

Launch: see ``run_teacher_distill_1cycle.sh``.
Outputs: ``$AE/runs/teacher-distill-1cycle/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Offline + flashinfer env vars must be set BEFORE any vllm/transformers import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP4", "0")
os.environ.setdefault("VLLM_ALLREDUCE_USE_FLASHINFER", "0")

AE = Path("/fsx/zzsamshi/a-evolve")
sys.path.insert(0, str(AE))

from agent_evolve.model.algorithms.mcgs.mutation import BaselineMutationProposer  # noqa: E402
from agent_evolve.model.algorithms.mcgs.search import MCGSSearch  # noqa: E402
from agent_evolve.model.api import TrainingEvolver  # noqa: E402
from agent_evolve.model.types import TrainingEvolveConfig  # noqa: E402


class EnableSynthMutator:
    """One-shot mutator that enables ``teacher_distill`` in ``train/pipeline.yaml``.

    MCGS still wants *some* mutation per cycle (otherwise fork would copy the
    seed unchanged). We flip the synth stage from ``enabled: false`` to
    ``enabled: true`` — that's our single mutation for this run.
    """

    def propose(self, parent, graph=None):  # noqa: ARG002
        from agent_evolve.model.types import (
            PatchOperation,
            WorkspaceMutation,
            WorkspacePatch,
        )

        return WorkspaceMutation(
            mutation_id="m-enable-synth",
            parent_node_id=parent.node_id,
            description="Enable teacher_distill stage",
            patch=WorkspacePatch(operations=[
                PatchOperation(
                    op="replace",
                    path="train/pipeline.yaml",
                    key_path=["stages", 0, "enabled"],
                    value=True,
                ),
            ]),
            mutation_type="training_recipe",
        )


def main() -> int:
    algo = MCGSSearch(mutator=EnableSynthMutator())
    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm=algo,
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=1,
            # Budget must cover: 120B load (~6m) + gen 500 prompts (~30-50m)
            # + SFT (~27m) + eval (~10m).
            trial_budget_seconds=7200,
        ),
        work_dir=AE / "runs" / "teacher-distill-1cycle",
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
