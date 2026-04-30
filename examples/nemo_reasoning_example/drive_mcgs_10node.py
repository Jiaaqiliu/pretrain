"""10-cycle MCGS campaign for the Kaggle Nemotron Reasoning Challenge.

Compact successor to ``drive_mcgs_20node.py`` — same idea, half the budget,
updated for the post-reorg pipeline layout and the new plugin registries
(``StageRegistry``, ``DataGenerator``, ``ModelAdapter`` — see §16 of
``TRAINDESIGN.md`` and ``INTEGRATION.md``).

Tree layout:

  cycles 1-5   fanout from root — each a GSPO variant starting from E-28:
      B1  gspo-fresh        group advantage, 100 steps, lr=3e-6, ε=(3e-4,4e-4)
      B2  loop-advantage    B1 + advantage_mode=loop
      B3  wider-clip        B1 + ε=(1e-3,1.5e-3)
      B4  domain-focus      B1 + domains=[bits,cipher,equations] at per_domain=250
      B5  more-rollouts     B1 + per_domain=250 (×2 across all 6 domains)

  cycles 6-9   depth-2 exploitation on UCT-selected winners:
      lr-down / step-up / G-up / DAPO

  cycle 10     LR-bag insurance (LRBagMutationProposer) on the incumbent

Pipeline stage order (post-reorg — see
``seed_workspaces/nemotron_reasoner/train/pipeline.yaml``):

    [0] solver_distill     (data-gen; default disabled)
    [1] teacher_distill    (stage.type=synth_generate; default disabled)
    [2] data_merge         (default disabled)
    [3] sft_warmup         (default enabled)
    [4] rl_gspo            (default disabled — we flip it on for all branches)

Every mutation therefore keys off ``stages[4]`` (RL) and flips
``stages[3].enabled=False`` (skip SFT — we start from E-28 and iterate on
the adapter via GSPO).

Single workspace (``seed_workspaces/nemotron_reasoner``), forked per cycle.
Sequential trials; each cycle uses all 8 GPUs.

Launch (foreground):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 AE_TRAIN_DDP=1 \\
      PYTHONPATH=/fsx/zzsamshi/a-evolve \\
      /fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \\
      examples/nemo_reasoning_example/drive_mcgs_10node.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

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

from agent_evolve.training.algorithms.mcgs.mutation import (  # noqa: E402
    LRBagMutationProposer,
)
from agent_evolve.training.algorithms.mcgs.search import MCGSSearch  # noqa: E402
from agent_evolve.training.algorithms.mcgs.selection import UCTSelector  # noqa: E402
from agent_evolve.training.api import TrainingEvolver  # noqa: E402
from agent_evolve.training.types import (  # noqa: E402
    PatchOperation,
    TrainingEvolveConfig,
    WorkspaceMutation,
    WorkspacePatch,
)

# Importing ``training.runners`` triggers the built-in ``@register_stage``
# side-effects for {sft, rl, synth_generate, solver_distill, data_merge,
# generate}. If you add a custom stage type via INTEGRATION.md §2, import
# that module here too — the decorator only fires on import.
import agent_evolve.training.runners  # noqa: F401,E402

# ── Starting adapter + data ─────────────────────────────────────────────

SEED_E28 = str(NAR / "experiments" / "E-28-iter3-noprm" / "adapter")
SEED_E28_NAME = "E-28-iter3-noprm"

# Stage indices in pipeline.yaml (see docstring).
SOLVER_DISTILL_IDX = 0
TEACHER_DISTILL_IDX = 1
DATA_MERGE_IDX = 2
SFT_IDX = 3
RL_IDX = 4


# ── Patch builders ──────────────────────────────────────────────────────


def _seed_ops() -> list[PatchOperation]:
    """Common to every branch: start from E-28, disable data + SFT stages,
    enable RL. ``override_seed_adapter`` forces the pipeline to train even
    though ``seed_adapter_path`` is set (otherwise ``run_trial`` would skip
    training and evaluate E-28 directly — the eval-only shortcut)."""
    def _disable(idx: int) -> PatchOperation:
        return PatchOperation(
            op="replace", path="train/pipeline.yaml",
            key_path=["stages", idx, "enabled"], value=False,
        )

    return [
        PatchOperation(
            op="replace", path="model/adapter.yaml",
            key_path=["seed_adapter_path"], value=SEED_E28,
        ),
        PatchOperation(
            op="replace", path="model/adapter.yaml",
            key_path=["seed_adapter_name"], value=SEED_E28_NAME,
        ),
        PatchOperation(
            op="replace", path="train/pipeline.yaml",
            key_path=["override_seed_adapter"], value=True,
        ),
        # Data-gen + SFT stages off by default for this campaign — we're
        # iterating on the already-trained E-28 adapter via GSPO only.
        _disable(SOLVER_DISTILL_IDX),
        _disable(TEACHER_DISTILL_IDX),
        _disable(DATA_MERGE_IDX),
        _disable(SFT_IDX),
        # RL on.
        PatchOperation(
            op="replace", path="train/pipeline.yaml",
            key_path=["stages", RL_IDX, "enabled"], value=True,
        ),
    ]


def _gspo_ops(
    *,
    max_steps: int = 100,
    advantage_mode: str = "group",
    eps_low: float = 3.0e-4,
    eps_high: float = 4.0e-4,
    lr: float = 3.0e-6,
    per_domain: int = 125,
    domains: list[str] | None = None,
    n_samples: int = 8,
    grad_accum: int = 8,
    dapo_token_level: bool = False,
    max_tokens: int = 2560,
    max_len: int = 2800,
    seed: int = 11,
) -> list[PatchOperation]:
    """Build the RL-stage patch block. Every key under ``stages[4]``."""
    if domains is None:
        domains = ["bits", "cipher", "equations", "gravity", "units", "numerals"]

    def key(k: str) -> list[Any]:
        return ["stages", RL_IDX, k]

    return [
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("max_steps"), value=max_steps),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("advantage_mode"), value=advantage_mode),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("eps_low"), value=eps_low),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("eps_high"), value=eps_high),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("lr"), value=lr),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("per_domain"), value=per_domain),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("domains"), value=domains),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("n_samples"), value=n_samples),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("grad_accum"), value=grad_accum),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("dapo_token_level"), value=dapo_token_level),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("max_tokens"), value=max_tokens),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("max_len"), value=max_len),
        PatchOperation(op="replace", path="train/pipeline.yaml", key_path=key("seed"), value=seed),
    ]


def _mk(desc: str, extra_ops: list[PatchOperation], *, mtype: str = "training_recipe") -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-{uuid.uuid4().hex[:8]}",
        parent_node_id="node-root",
        description=desc,
        patch=WorkspacePatch(operations=_seed_ops() + extra_ops),
        mutation_type=mtype,  # type: ignore[arg-type]
    )


# ── 5 fanout branches (cycles 1-5) ──────────────────────────────────────


def b1_gspo_fresh() -> WorkspaceMutation:
    return _mk("B1 gspo-fresh (group adv, 100 steps)", _gspo_ops())


def b2_loop_advantage() -> WorkspaceMutation:
    return _mk(
        "B2 loop-advantage (LOOP z-score)",
        _gspo_ops(advantage_mode="loop"),
        mtype="loss",
    )


def b3_wider_clip() -> WorkspaceMutation:
    return _mk(
        "B3 wider-clip (ε=1e-3/1.5e-3)",
        _gspo_ops(eps_low=1.0e-3, eps_high=1.5e-3),
        mtype="loss",
    )


def b4_domain_focus() -> WorkspaceMutation:
    return _mk(
        "B4 domain-focus (bits+cipher+equations, 250/domain)",
        _gspo_ops(domains=["bits", "cipher", "equations"], per_domain=250),
        mtype="reward",
    )


def b5_more_rollouts() -> WorkspaceMutation:
    return _mk(
        "B5 more-rollouts (per_domain=250, all 6 domains)",
        _gspo_ops(per_domain=250),
        mtype="rollout",
    )


BRANCHES = [b1_gspo_fresh, b2_loop_advantage, b3_wider_clip,
            b4_domain_focus, b5_more_rollouts]


# ── Depth-2 moves (cycles 6-9) ──────────────────────────────────────────
#
# Each move expresses a delta on the parent node's workspace. MCGS's
# ``workspace.fork`` copies the parent and applies ONLY this delta, so
# depth-2 moves are intentionally tiny.


def d2_lr_down(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-lr-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="d2 lr-down 3e-6 → 1.5e-6",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace", path="train/pipeline.yaml",
            key_path=["stages", RL_IDX, "lr"], value=1.5e-6,
        )]),
        mutation_type="training_recipe",
    )


def d2_step_up(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-steps-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="d2 max_steps 100 → 150",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace", path="train/pipeline.yaml",
            key_path=["stages", RL_IDX, "max_steps"], value=150,
        )]),
        mutation_type="training_recipe",
    )


def d2_g_up(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-g-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="d2 n_samples 8 → 12 (more rollouts/prompt)",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace", path="train/pipeline.yaml",
            key_path=["stages", RL_IDX, "n_samples"], value=12,
        )]),
        mutation_type="rollout",
    )


def d2_dapo(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-dapo-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="d2 DAPO token-level",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace", path="train/pipeline.yaml",
            key_path=["stages", RL_IDX, "dapo_token_level"], value=True,
        )]),
        mutation_type="loss",
    )


DEPTH2_MOVES = [d2_lr_down, d2_step_up, d2_g_up, d2_dapo]


# ── Selector: fanout-then-UCT ───────────────────────────────────────────


class FanoutThenUCTSelector:
    """First ``fanout`` cycles force root as parent (so we get distinct
    sibling branches); subsequent cycles delegate to UCT."""

    def __init__(self, fanout: int = 5, c_init: float = 1.4) -> None:
        self.fanout = fanout
        self._uct = UCTSelector(c_init=c_init)

    def select(self, graph, *, cycle: int):
        root = graph.root()
        assert root is not None
        direct_children = [
            n for n in graph.nodes.values() if n.parent_id == root.node_id
        ]
        if len(direct_children) < self.fanout:
            return root
        return self._uct.select(graph, cycle=cycle)


# ── Campaign mutator ────────────────────────────────────────────────────


class CampaignMutator:
    """Cycle-aware mutator. Stateful — remembers which depth-2 moves have
    been tried per parent node, so the same move doesn't fire twice on the
    same parent."""

    def __init__(self) -> None:
        self._cycle = 0
        self._tried_depth2: set[tuple[str, str]] = set()
        # LR-bag insurance for the final cycle — one LR the other moves
        # haven't covered, chosen from a small curated bag.
        self._lr_bag = LRBagMutationProposer(bag=(1.5e-6, 6e-6, 5e-6, 2e-6))

    def propose(self, parent: Any, graph: Any = None) -> WorkspaceMutation:  # noqa: ARG002
        self._cycle += 1
        cycle = self._cycle

        if cycle <= 5:
            mutation = BRANCHES[cycle - 1]()
            mutation.parent_node_id = parent.node_id
            return mutation

        if cycle <= 9:
            return self._pick_depth2(parent)

        # cycle 10: LR-bag insurance on the UCT-selected incumbent.
        return self._lr_bag.propose(parent)

    def _pick_depth2(self, parent: Any) -> WorkspaceMutation:
        for move in DEPTH2_MOVES:
            key = (parent.node_id, move.__name__)
            if key in self._tried_depth2:
                continue
            self._tried_depth2.add(key)
            return move(parent)
        # All moves tried on this parent — fall back to an LR perturbation.
        return self._lr_bag.propose(parent)


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    mutator = CampaignMutator()
    selector = FanoutThenUCTSelector(fanout=5, c_init=1.4)
    algo = MCGSSearch(mutator=mutator, selector=selector)

    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm=algo,
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=10,
            trial_budget_seconds=14400,  # 4h/cycle hard cap
        ),
        work_dir=AE / "runs" / "mcgs-10node",
    )
    result = evolver.run(cycles=10)

    print("\n=== Campaign Complete ===")
    print(f"cycles_completed:  {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric:       {result.best_metric}")
    print(f"graph_path:        {result.graph_path}")
    print(f"report_path:       {result.report_path}")
    for entry in result.topk:
        print(
            f"  topk: node={entry.node_id} branch={entry.branch_id} "
            f"metric={entry.metric} reward={entry.reward}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
