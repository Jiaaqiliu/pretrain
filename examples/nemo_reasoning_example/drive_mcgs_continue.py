"""Continuation campaign: start from cycle-1's 51.31% B1 adapter and
explore data + algorithm knobs to push dev higher.

Design lessons baked in from the prior run:
  - Every branch EXPLICITLY sets stages[1].enabled=False (GSPO-only) so we
    never accidentally pick up SFT from a parent workspace that might have
    it enabled. The prior run's cycle-7 hang was a UCT-picked-root trap.
  - Every branch EXPLICITLY patches data/sources.yaml ONLY when it needs
    to add a file whose schema matches render_hf_dataset (prompt_rendered +
    completion). Otherwise leave sources.yaml alone (seed has one valid
    file: short_correct.jsonl).
  - Starting adapter = the cycle-1 B1 adapter (51.31%), not E-28. Loaded
    via seed_adapter_path + override_seed_adapter=True.

20 cycles, all depth-1 from root. Each branch varies:
  - rollout domain mix / per_domain
  - GSPO knobs: max_steps, lr, eps, advantage_mode, dapo
  - SFT first (when schema is safe)

Assumes AE_TRAIN_DDP=1 at launch time → DDP through ddp_launcher.
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

from agent_evolve.training.algorithms.mcgs.search import MCGSSearch  # noqa: E402
from agent_evolve.training.api import TrainingEvolver  # noqa: E402
from agent_evolve.training.types import (  # noqa: E402
    PatchOperation,
    TrainingEvolveConfig,
    WorkspaceMutation,
    WorkspacePatch,
)

# ── Starting adapter: cycle-1 B1 from the prior campaign (51.31%) ───────

PRIOR_BEST = str(
    AE / "runs" / "mcgs-20node" / "nodes" / "node-4566d52083"
    / "workspace" / "checkpoints" / "adapters" / "rl_gspo"
)
PRIOR_BEST_NAME = "continue-B1-5131"
# Fallback to E-28 if the prior adapter somehow isn't present.
E28 = str(NAR / "experiments" / "E-28-iter3-noprm" / "adapter")
START_ADAPTER = PRIOR_BEST if Path(PRIOR_BEST).is_dir() else E28
START_NAME = PRIOR_BEST_NAME if Path(PRIOR_BEST).is_dir() else "E-28-iter3-noprm"

SHORT_CORRECT = str(NAR / "data" / "sft" / "short_correct.jsonl")


# ── Helpers to build mutation patches ───────────────────────────────────

def _seed_ops() -> list[PatchOperation]:
    return [
        PatchOperation(op="replace", path="model/adapter.yaml",
                       key_path=["seed_adapter_path"], value=START_ADAPTER),
        PatchOperation(op="replace", path="model/adapter.yaml",
                       key_path=["seed_adapter_name"], value=START_NAME),
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["override_seed_adapter"], value=True),
        # Disable every non-SFT/non-RL stage (the new workspace has
        # solver_distill + data_merge stages that we don't touch yet).
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["stages", 0, "enabled"], value=False),  # solver_distill
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["stages", 1, "enabled"], value=False),  # teacher_distill
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["stages", 2, "enabled"], value=False),  # data_merge
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["stages", 3, "enabled"], value=False),  # sft_warmup
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["stages", 4, "enabled"], value=True),   # rl_gspo
    ]


def _gspo_ops(
    *,
    per_domain: int = 125,
    domains: list[str] | None = None,
    n_samples: int = 8,
    max_steps: int = 100,
    grad_accum: int = 8,
    lr: float = 3.0e-6,
    eps_low: float = 3.0e-4,
    eps_high: float = 4.0e-4,
    advantage_mode: str = "group",
    dapo_token_level: bool = False,
    max_tokens: int = 2560,
    max_len: int = 2800,
    seed: int = 11,
) -> list[PatchOperation]:
    """Build the GSPO stage patch block. Stage index 4 in the new pipeline."""
    if domains is None:
        domains = ["bits", "cipher", "equations", "gravity", "units", "numerals"]
    key = lambda k: ["stages", 4, k]  # noqa: E731
    return [
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("max_steps"),  value=max_steps),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("advantage_mode"), value=advantage_mode),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("eps_low"),  value=eps_low),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("eps_high"), value=eps_high),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("lr"),       value=lr),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("per_domain"), value=per_domain),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("domains"),  value=domains),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("n_samples"), value=n_samples),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("grad_accum"), value=grad_accum),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("max_tokens"), value=max_tokens),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("max_len"),  value=max_len),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("seed"),     value=seed),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("dapo_token_level"), value=dapo_token_level),
    ]


def _mk(desc: str, extra_ops: list[PatchOperation], *, mtype: str = "training_recipe") -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-{uuid.uuid4().hex[:8]}",
        parent_node_id="node-root",
        description=desc,
        patch=WorkspacePatch(operations=_seed_ops() + extra_ops),
        mutation_type=mtype,  # type: ignore[arg-type]
    )


# ── 20 branches to explore ──────────────────────────────────────────────

def b01() -> WorkspaceMutation:
    # Calibration — reproduce 51.31% on the cycle-1 B1 start adapter.
    return _mk("B01 reproduce-B1 (calib)", _gspo_ops())

def b02() -> WorkspaceMutation:
    return _mk("B02 LOOP advantage",
               _gspo_ops(advantage_mode="loop"), mtype="loss")

def b03() -> WorkspaceMutation:
    return _mk("B03 domain advantage",
               _gspo_ops(advantage_mode="domain"), mtype="loss")

def b04() -> WorkspaceMutation:
    return _mk("B04 wider clip 1e-3/1.5e-3",
               _gspo_ops(eps_low=1.0e-3, eps_high=1.5e-3), mtype="loss")

def b05() -> WorkspaceMutation:
    return _mk("B05 DAPO token-level",
               _gspo_ops(dapo_token_level=True), mtype="loss")

def b06() -> WorkspaceMutation:
    return _mk("B06 lr-down 1.5e-6", _gspo_ops(lr=1.5e-6))

def b07() -> WorkspaceMutation:
    return _mk("B07 lr-up 6e-6", _gspo_ops(lr=6e-6))

def b08() -> WorkspaceMutation:
    return _mk("B08 more rollouts per_domain=250", _gspo_ops(per_domain=250))

def b09() -> WorkspaceMutation:
    return _mk("B09 more rollouts per_domain=500 (bits+cipher+eq)",
               _gspo_ops(per_domain=500,
                         domains=["bits", "cipher", "equations"]))

def b10() -> WorkspaceMutation:
    return _mk("B10 G=16 (double rollouts/prompt)",
               _gspo_ops(n_samples=16))

def b11() -> WorkspaceMutation:
    return _mk("B11 longer training 150 steps", _gspo_ops(max_steps=150))

def b12() -> WorkspaceMutation:
    return _mk("B12 shorter training 60 steps (early-stop style)",
               _gspo_ops(max_steps=60, lr=6e-6))

def b13() -> WorkspaceMutation:
    return _mk("B13 data: weak-domain focus (bits+cipher+eq)",
               _gspo_ops(per_domain=250, domains=["bits", "cipher", "equations"]))

def b14() -> WorkspaceMutation:
    return _mk("B14 data: strong-domain protect (gravity+units+numerals)",
               _gspo_ops(per_domain=250, domains=["gravity", "units", "numerals"]))

def b15() -> WorkspaceMutation:
    return _mk("B15 LOOP + wider clip fusion",
               _gspo_ops(advantage_mode="loop", eps_low=1e-3, eps_high=1.5e-3),
               mtype="fusion")

def b16() -> WorkspaceMutation:
    return _mk("B16 LOOP + lr-down fusion",
               _gspo_ops(advantage_mode="loop", lr=1.5e-6),
               mtype="fusion")

def b17() -> WorkspaceMutation:
    # Seed diversity — different sampling seed for rollouts.
    return _mk("B17 seed-43 diversity", _gspo_ops(seed=43))

def b18() -> WorkspaceMutation:
    # DAPO + LOOP combo.
    return _mk("B18 DAPO + LOOP fusion",
               _gspo_ops(dapo_token_level=True, advantage_mode="loop"),
               mtype="fusion")

def b19() -> WorkspaceMutation:
    # Very-long rollouts — up max_tokens.
    return _mk("B19 long rollouts max_tokens=3584",
               _gspo_ops(max_tokens=3584, max_len=3800))

def b20() -> WorkspaceMutation:
    # SFT preamble + GSPO. Adds short_correct as sources, enables SFT.
    ops = _seed_ops()
    # Enable SFT stage (index 3) with tight budget.
    ops = [o for o in ops if not (o.path == "train/pipeline.yaml" and o.key_path == ["stages", 3, "enabled"])]
    ops.extend([
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["stages", 3, "enabled"], value=True),
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["stages", 3, "max_steps"], value=10),
        PatchOperation(op="replace", path="train/pipeline.yaml",
                       key_path=["stages", 3, "epochs"], value=1),
        # Ensure sources.yaml has only short_correct.jsonl (the known-good file).
        PatchOperation(op="replace", path="data/sources.yaml",
                       key_path=["sources"], value=[
                           {"path": SHORT_CORRECT, "split": "train", "format": "jsonl"},
                       ]),
    ])
    ops.extend(_gspo_ops(max_steps=80))
    return WorkspaceMutation(
        mutation_id=f"m-{uuid.uuid4().hex[:8]}",
        parent_node_id="node-root",
        description="B20 SFT10 + GSPO80",
        patch=WorkspacePatch(operations=ops),
        mutation_type="pipeline",
    )


BRANCHES = [b01, b02, b03, b04, b05, b06, b07, b08, b09, b10,
            b11, b12, b13, b14, b15, b16, b17, b18, b19, b20]


# ── Selector + mutator ──────────────────────────────────────────────────


class RootOnlySelector:
    """All 20 cycles branch off root — no depth-2 traps."""

    def select(self, graph, *, cycle: int):  # noqa: ARG002
        root = graph.root()
        assert root is not None
        return root


class OrderedBranchMutator:
    """Cycle N dispatches to BRANCHES[N-1]. Deterministic, reproducible."""

    def __init__(self):
        self._cycle = 0

    def propose(self, parent, graph=None):  # noqa: ARG002
        self._cycle += 1
        idx = (self._cycle - 1) % len(BRANCHES)
        mut = BRANCHES[idx]()
        mut.parent_node_id = parent.node_id
        return mut


def main() -> int:
    algo = MCGSSearch(
        mutator=OrderedBranchMutator(),
        selector=RootOnlySelector(),
    )
    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm=algo,
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=20,
            trial_budget_seconds=14400,  # 4hr/cycle cap
        ),
        work_dir=AE / "runs" / "mcgs-continue",
    )
    result = evolver.run(cycles=20)
    print("\n=== Continuation Campaign Complete ===")
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric: {result.best_metric}")
    for entry in result.topk:
        print(f"  topk: node={entry.node_id} branch={entry.branch_id} "
              f"metric={entry.metric} reward={entry.reward}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
