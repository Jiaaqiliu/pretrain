"""20-cycle MCGS campaign for the Kaggle Nemotron Reasoning Challenge.

Tree layout (from the approved plan):

  cycles 1-6  fanout layer (parent = root, each branch from E-28):
      B1  gspo-fresh (calibration): 100 steps, group advantage, ε=(3e-4,4e-4), lr=3e-6
      B2  loop-advantage:           B1 + advantage_mode=loop
      B3  wider-clip:                B1 + ε=(1e-3,1.5e-3)
      B4  domain-prm:                B1 + domain-PRM (bits/cipher/eq weighted 1, rest 0)
      B5  more-prompts:              B1 + per_domain=500, n_samples=8
      B6  sft-expand:                SFT 30 steps on existing short_correct+synth_prompts_harvest, then GSPO 80

  cycles 7-14  depth-2 exploitation (UCT + depth-2 mutator):
      lr-down / step-up / G-up / DAPO / style-guard / cross-branch
      / advantage-swap / data-mix add self-distill

  cycles 15-20  fusion + exploit:
      cycle 15 fusion across top-2, cycle 16-18 exploit-depth-3,
      cycle 19 seed-repeat incumbent, cycle 20 LRBag insurance

Single workspace (``seed_workspaces/nemotron_reasoner``), forked per cycle.
Sequential trials; each cycle is allowed to use all 8 GPUs.
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

# ── Constants: starting adapters + data ─────────────────────────────────

SEED_E28 = str(NAR / "experiments" / "E-28-iter3-noprm" / "adapter")
SEED_E33 = str(NAR / "experiments" / "E-33-iter3-loop" / "adapter")

DATA_SHORT_CORRECT = str(NAR / "data" / "sft" / "short_correct.jsonl")
DATA_SYNTH_HARVEST = str(NAR / "data" / "sft" / "synth_prompts_harvest.jsonl")
DATA_SYNTH_SHORT_CORRECT = str(
    NAR / "data" / "sft" / "synth_harvest_short_correct.jsonl"
)

# ── Branch patch builders ───────────────────────────────────────────────
#
# Each builder returns a list of PatchOperation. All 6 base branches start
# from E-28 via ``model/adapter.yaml::seed_adapter_path`` and flip
# ``override_seed_adapter=True`` on the pipeline so the trial actually trains.


def _seed_from_e28_ops() -> list[PatchOperation]:
    return [
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
    ]


def _base_gspo_ops() -> list[PatchOperation]:
    """Common B1 knobs: GSPO iter, group advantage, 100 steps, lr=3e-6.

    Stage indexing in pipeline.yaml:
      [0] = teacher_distill (synth_generate, default disabled)
      [1] = sft_warmup       (sft, default enabled)
      [2] = rl_gspo          (rl, default disabled)
    """
    return [
        # SFT off — we're iterating on the adapter, not training from base.
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 1, "enabled"],
            value=False,
        ),
        # GSPO on.
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "enabled"],
            value=True,
        ),
        # Calibration recipe matching E-28's iter3.
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "max_steps"],
            value=100,
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
            key_path=["stages", 2, "lr"],
            value=3.0e-6,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "per_domain"],
            value=125,  # ~750 prompts total across 6 domains
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "domains"],
            value=["bits", "cipher", "equations", "gravity", "units", "numerals"],
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "n_samples"],
            value=8,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "grad_accum"],
            value=8,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "max_tokens"],
            value=2560,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "max_len"],
            value=2800,
        ),
    ]


# ── The 6 base branches ─────────────────────────────────────────────────


def b1_gspo_fresh() -> WorkspaceMutation:
    ops = _seed_from_e28_ops() + _base_gspo_ops()
    return WorkspaceMutation(
        mutation_id=f"m-B1-cal-{uuid.uuid4().hex[:6]}",
        parent_node_id="node-root",
        description="B1 gspo-fresh calibration (E-28 + 100 steps group adv)",
        patch=WorkspacePatch(operations=ops),
        mutation_type="training_recipe",
    )


def b2_loop_advantage() -> WorkspaceMutation:
    ops = _seed_from_e28_ops() + _base_gspo_ops()
    ops.append(
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "advantage_mode"],
            value="loop",
        )
    )
    return WorkspaceMutation(
        mutation_id=f"m-B2-loop-{uuid.uuid4().hex[:6]}",
        parent_node_id="node-root",
        description="B2 loop-advantage (LOOP z-score, E-33 reproduction from E-28)",
        patch=WorkspacePatch(operations=ops),
        mutation_type="training_recipe",
    )


def b3_wider_clip() -> WorkspaceMutation:
    ops = _seed_from_e28_ops() + _base_gspo_ops()
    ops.extend([
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "eps_low"],
            value=1.0e-3,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "eps_high"],
            value=1.5e-3,
        ),
    ])
    return WorkspaceMutation(
        mutation_id=f"m-B3-wide-eps-{uuid.uuid4().hex[:6]}",
        parent_node_id="node-root",
        description="B3 wider-clip (ε=1e-3/1.5e-3)",
        patch=WorkspacePatch(operations=ops),
        mutation_type="loss",
    )


def b4_domain_prm() -> WorkspaceMutation:
    # "domain-PRM" = restrict to the weak domains only, rely on group advantage
    # to focus lift where headroom is (bits/cipher/equations).
    ops = _seed_from_e28_ops() + _base_gspo_ops()
    ops.extend([
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "domains"],
            value=["bits", "cipher", "equations"],
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "per_domain"],
            value=250,  # ~750 prompts, same total
        ),
    ])
    return WorkspaceMutation(
        mutation_id=f"m-B4-domain-prm-{uuid.uuid4().hex[:6]}",
        parent_node_id="node-root",
        description="B4 domain-PRM (bits+cipher+equations only)",
        patch=WorkspacePatch(operations=ops),
        mutation_type="reward",
    )


def b5_more_prompts() -> WorkspaceMutation:
    ops = _seed_from_e28_ops() + _base_gspo_ops()
    ops.append(
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "per_domain"],
            value=250,  # ~1500 total across 6 domains
        )
    )
    return WorkspaceMutation(
        mutation_id=f"m-B5-more-{uuid.uuid4().hex[:6]}",
        parent_node_id="node-root",
        description="B5 more-prompts (1500 rollouts)",
        patch=WorkspacePatch(operations=ops),
        mutation_type="rollout",
    )


def b6_sft_expand() -> WorkspaceMutation:
    # Keep SFT ON with an expanded data mix, then run short GSPO afterward.
    ops = _seed_from_e28_ops()
    ops.extend([
        # Add the synth-prompts harvest to data/sources.yaml.
        PatchOperation(
            op="replace",
            path="data/sources.yaml",
            key_path=["sources"],
            value=[
                {"path": DATA_SHORT_CORRECT, "split": "train", "format": "jsonl"},
                {"path": DATA_SYNTH_HARVEST, "split": "train", "format": "jsonl"},
            ],
        ),
        # SFT: 30 steps, grad_accum=8.
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 1, "enabled"],
            value=True,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 1, "max_steps"],
            value=30,
        ),
        PatchOperation(
            op="replace",
            path="train/batching.yaml",
            key_path=["grad_accum"],
            value=8,
        ),
        # GSPO: 80 steps after SFT.
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "enabled"],
            value=True,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "max_steps"],
            value=80,
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
            key_path=["stages", 2, "lr"],
            value=3.0e-6,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "per_domain"],
            value=125,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "domains"],
            value=["bits", "cipher", "equations", "gravity", "units", "numerals"],
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "n_samples"],
            value=8,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "grad_accum"],
            value=8,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "max_tokens"],
            value=2560,
        ),
        PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "max_len"],
            value=2800,
        ),
    ])
    return WorkspaceMutation(
        mutation_id=f"m-B6-sft-{uuid.uuid4().hex[:6]}",
        parent_node_id="node-root",
        description="B6 sft-expand (SFT30 + GSPO80 on expanded data mix)",
        patch=WorkspacePatch(operations=ops),
        mutation_type="pipeline",
    )


BRANCHES = [b1_gspo_fresh, b2_loop_advantage, b3_wider_clip,
            b4_domain_prm, b5_more_prompts, b6_sft_expand]


# ── Depth-2 moves (cycles 7-14) ─────────────────────────────────────────
#
# Each move takes a parent node's patches (applied to its workspace on fork)
# and adds a small delta. Because MCGS.fork already copies the parent's
# workspace (with all prior patches baked in), depth-2 mutations only need
# to express the delta.

def depth2_lr_down(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-lr-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="lr-down 3e-6 → 1.5e-6",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "lr"],
            value=1.5e-6,
        )]),
        mutation_type="training_recipe",
    )


def depth2_step_up(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-steps-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="max_steps 100 → 150",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "max_steps"],
            value=150,
        )]),
        mutation_type="training_recipe",
    )


def depth2_g_up(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-g-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="n_samples 8 → 12 (more rollouts per prompt)",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "n_samples"],
            value=12,
        )]),
        mutation_type="rollout",
    )


def depth2_dapo(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-dapo-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="DAPO token-level",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "dapo_token_level"],
            value=True,
        )]),
        mutation_type="loss",
    )


def depth2_advantage_domain(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-adv-domain-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="advantage_mode → domain (across-domain z-score)",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "advantage_mode"],
            value="domain",
        )]),
        mutation_type="loss",
    )


def depth2_advantage_loop(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-adv-loop-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="advantage_mode → loop (LOO)",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace",
            path="train/pipeline.yaml",
            key_path=["stages", 2, "advantage_mode"],
            value="loop",
        )]),
        mutation_type="loss",
    )


def depth2_length_penalty(parent) -> WorkspaceMutation:
    # Style guard: penalize rollouts over 2500 tokens (matches CLAUDE.md cap).
    return WorkspaceMutation(
        mutation_id=f"m-d2-lenpen-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="length_penalty λ=0.5 (style guard)",
        patch=WorkspacePatch(operations=[
            PatchOperation(
                op="replace",
                path="train/pipeline.yaml",
                key_path=["stages", 2, "length_penalty_lambda"],
                value=0.5,
            ),
            PatchOperation(
                op="replace",
                path="train/pipeline.yaml",
                key_path=["stages", 2, "length_penalty_cap"],
                value=2500,
            ),
        ]),
        mutation_type="reward",
    )


def depth2_data_mix_add_synth(parent) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id=f"m-d2-mix-{uuid.uuid4().hex[:6]}",
        parent_node_id=parent.node_id,
        description="data/sources.yaml += synth_harvest_short_correct",
        patch=WorkspacePatch(operations=[PatchOperation(
            op="replace",
            path="data/sources.yaml",
            key_path=["sources"],
            value=[
                {"path": DATA_SHORT_CORRECT, "split": "train", "format": "jsonl"},
                {"path": DATA_SYNTH_HARVEST, "split": "train", "format": "jsonl"},
                {"path": DATA_SYNTH_SHORT_CORRECT, "split": "train", "format": "jsonl"},
            ],
        )]),
        mutation_type="data_mix",
    )


DEPTH2_MOVES = [
    depth2_lr_down,
    depth2_step_up,
    depth2_g_up,
    depth2_dapo,
    depth2_advantage_loop,
    depth2_advantage_domain,
    depth2_length_penalty,
    depth2_data_mix_add_synth,
]


# ── Campaign mutator ────────────────────────────────────────────────────


class CampaignMutator:
    """Cycle-aware mutator that drives the 20-node plan.

    Stateful: remembers which branches + depth-2 moves have been tried.
    """

    def __init__(self) -> None:
        self._cycle = 0
        self._tried_depth2: set[tuple[str, str]] = set()  # (parent_node_id, move_name)
        self._lr_bag = LRBagMutationProposer(bag=(1.5e-6, 6e-6, 2e-6, 5e-6))

    def propose(self, parent: Any, graph: Any = None) -> WorkspaceMutation:
        self._cycle += 1
        cycle = self._cycle

        # Cycles 1-6: fanout (parent is root, enforced by RootFanoutSelector).
        if cycle <= 6:
            mutation = BRANCHES[cycle - 1]()
            mutation.parent_node_id = parent.node_id  # keep in sync
            return mutation

        # Cycles 7-14: depth-2 moves on the best branch(es).
        if cycle <= 14:
            return self._pick_depth2(parent, graph)

        # Cycle 15+: fusion / exploit / seed-repeat / LR-bag insurance.
        if cycle == 15:
            return self._pick_depth2(parent, graph, prefer_new_parent=True)
        if cycle in (16, 17, 18):
            return self._pick_depth2(parent, graph)
        if cycle == 19:
            # Seed-repeat: same mutation as the current incumbent, but bump
            # the seed knob so we get a second point for noise calibration.
            return self._seed_repeat(parent)
        # cycle 20: LR-bag insurance (one LR MCGS hasn't tried yet).
        return self._lr_bag.propose(parent, graph=graph)

    def _pick_depth2(
        self, parent: Any, graph: Any, *, prefer_new_parent: bool = False
    ) -> WorkspaceMutation:
        del graph, prefer_new_parent  # UCT already picked the parent for us
        for move in DEPTH2_MOVES:
            key = (parent.node_id, move.__name__)
            if key in self._tried_depth2:
                continue
            self._tried_depth2.add(key)
            return move(parent)
        # All moves tried on this parent — fall back to an LR perturbation.
        return self._lr_bag.propose(parent, graph=None)

    def _seed_repeat(self, parent: Any) -> WorkspaceMutation:
        return WorkspaceMutation(
            mutation_id=f"m-seed-repeat-{uuid.uuid4().hex[:6]}",
            parent_node_id=parent.node_id,
            description="seed-repeat: bump stage seed for noise-floor point",
            patch=WorkspacePatch(operations=[PatchOperation(
                op="replace",
                path="train/pipeline.yaml",
                key_path=["stages", 2, "seed"],
                value=43,  # default is 11 in pipeline.yaml; bumping avoids cache
            )]),
            mutation_type="debug",
        )


# ── Selector: fanout for cycles 1-6, then UCT ────────────────────────────


class FanoutThenUCTSelector:
    def __init__(self, fanout: int = 6, c_init: float = 1.4) -> None:
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


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    mutator = CampaignMutator()
    selector = FanoutThenUCTSelector(fanout=6, c_init=1.4)
    algo = MCGSSearch(mutator=mutator, selector=selector)

    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm=algo,
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=20,
            trial_budget_seconds=18000,  # 5hr/cycle hard cap
        ),
        work_dir=AE / "runs" / "mcgs-20node",
    )
    result = evolver.run(cycles=20)

    print("\n=== Campaign Complete ===")
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
