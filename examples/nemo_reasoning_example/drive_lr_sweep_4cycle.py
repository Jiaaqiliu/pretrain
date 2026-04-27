"""4-cycle real-SFT auto-Kaggle run, MCGS-driven.

Fans out 4 siblings from root, each with a different LR from
``LRBagMutationProposer.bag``. Each cycle:

  1. Fork candidate workspace.
  2. Train rank-16 LoRA on short_correct.jsonl (~27 min at grad_accum=32).
  3. Evaluate on 951-row Kaggle dev with vLLM (~8 min).
  4. MCGS absorbs (metric, error buckets, cost), promotes best-of-seen.

No Kaggle submission. Incumbent materialized into the workspace's
``evolution/incumbent/`` for later inspection.

Launch: see ``examples/nemo_reasoning_example/run_lr_sweep_4cycle.sh``.
Outputs are written to ``$AE/runs/lr-sweep-4cycle/``:
  * ``logs/run.log`` — stdout/stderr
  * ``nemotron_reasoner/evolution/reports/cycle_NNNN.json`` — per-cycle MCGS report
  * ``nemotron_reasoner/evolution/mcgs_graph.json`` — full DAG
  * ``nodes/node-*/workspace/checkpoints/adapters/sft_warmup/`` — trained LoRAs
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

from agent_evolve.training.algorithms.mcgs.mutation import LRBagMutationProposer  # noqa: E402
from agent_evolve.training.algorithms.mcgs.search import MCGSSearch  # noqa: E402
from agent_evolve.training.api import TrainingEvolver  # noqa: E402
from agent_evolve.training.types import TrainingEvolveConfig  # noqa: E402


class RootFanoutSelector:
    """Forces the first ``fanout`` cycles to pick root as parent.

    The default UCTSelector would, once the first child has a metric, pick
    that child as next parent — collapsing the tree into a depth-1 chain.
    We want 4 distinct siblings of root so MCGS sees 4 independent scores.
    """

    def __init__(self, fanout: int = 4) -> None:
        self.fanout = fanout

    def select(self, graph, *, cycle: int):  # noqa: ARG002
        root = graph.root()
        assert root is not None
        direct_children = [
            n for n in graph.nodes.values() if n.parent_id == root.node_id
        ]
        if len(direct_children) < self.fanout:
            return root
        # Fallback: pick the current best direct child (for cycles > fanout).
        return max(
            direct_children,
            key=lambda n: (n.mean_reward, n.metric or float("-inf")),
        )


def main() -> int:
    algo = MCGSSearch(
        mutator=LRBagMutationProposer(bag=(1e-4, 5e-5, 3e-5, 1e-5)),
        selector=RootFanoutSelector(fanout=4),
    )
    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm=algo,
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=4,
            trial_budget_seconds=3600,  # 60 min/cycle hard cap
        ),
        work_dir=AE / "runs" / "lr-sweep-4cycle",
    )
    result = evolver.run(cycles=4)
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
