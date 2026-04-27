"""PR5 acceptance: invalid trial creates node with negative reward."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.training.algorithms.mcgs import MCGSSearch
from agent_evolve.training.loop import TrainingEvolutionLoop
from agent_evolve.training.types import (
    CheckpointRef,
    EvalMetrics,
    ErrorBuckets,
    TrainingTrialResult,
    ValidityReport,
)
from agent_evolve.training.workspace import TrainingWorkspace

from .fakes import FakeBenchmark


class _FailingBackend:
    name = "fail"

    def run_trial(self, workspace, node, budget, benchmark):
        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=str(workspace.root),
            status="train_failed",
            checkpoint=None,
            eval_metrics=None,
            error_buckets=ErrorBuckets(counts={}),
            validity=ValidityReport(is_valid=False, hard_fail_reason="train_failed"),
        )


def test_invalid_trial_has_negative_reward(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    algo = MCGSSearch()
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=algo,
        backend=_FailingBackend(),
        work_dir=tmp_path / "work",
    )
    loop.run(cycles=1)
    non_root = [n for n in algo.graph.nodes.values() if n.node_id != "node-root"]
    assert len(non_root) == 1
    node = non_root[0]
    assert node.is_valid is False
    assert node.reward is not None and node.reward < 0
    assert node.trial_status == "train_failed"
