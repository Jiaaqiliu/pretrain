"""Real 4-cycle LR sweep for posttrain_bench SFT.

Each cycle forks a fresh sibling from root, trains full-param SFT on
DeepSeek-R1-Distill-Qwen-1.5B with one of [1e-4, 5e-5, 3e-5, 1e-5],
then evaluates safety refusal rate on the eval split.

Launch: see ``examples/posttrain_bench/run_lr_sweep_real.sh``..
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

AE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AE))

from agent_evolve.model.algorithms.mcgs.mutation import LRBagMutationProposer  # noqa: E402
from agent_evolve.model.algorithms.mcgs.search import MCGSSearch  # noqa: E402
from agent_evolve.model.api import TrainingEvolver  # noqa: E402
from agent_evolve.model.types import TrainingEvolveConfig  # noqa: E402


class RootFanoutSelector:
    """Forces the first ``fanout`` cycles to pick root as parent."""

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
        workspace=AE / "seed_workspaces" / "posttrain_bench",
        benchmark="posttrain_bench",
        algorithm=algo,
        backend="h200_single_node",
        config=TrainingEvolveConfig(
            smoke=False,
            max_cycles=4,
            trial_budget_seconds=7200,
        ),
        work_dir=AE / "runs" / "posttrain-lr-sweep-real",
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
