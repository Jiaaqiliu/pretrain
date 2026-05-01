"""PR5 acceptance: low-scoring valid trial is kept in the graph."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.model.algorithms.mcgs import MCGSSearch
from agent_evolve.model.loop import TrainingEvolutionLoop
from agent_evolve.model.types import (
    CheckpointRef,
    EvalMetrics,
    ErrorBuckets,
    TrainingTrialResult,
    ValidityReport,
)
from agent_evolve.model.workspace import TrainingWorkspace

from .fakes import FakeBenchmark


class _LowScoreBackend:
    name = "low"

    def __init__(self) -> None:
        self.calls = 0

    def run_trial(self, workspace, node, budget, benchmark):
        self.calls += 1
        ckpt_dir = Path(workspace.root) / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=str(workspace.root),
            status="success",
            checkpoint=CheckpointRef(name="c", path=str(ckpt_dir)),
            eval_metrics=EvalMetrics(
                primary_metric_name="fake_metric", primary_metric_value=0.01
            ),
            error_buckets=ErrorBuckets(counts={}),
            validity=ValidityReport(is_valid=True),
        )


def test_low_score_node_kept(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    algo = MCGSSearch()
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=algo,
        backend=_LowScoreBackend(),
        work_dir=tmp_path / "work",
    )
    loop.run(cycles=2)
    non_root = [n for n in algo.graph.nodes.values() if n.node_id != "node-root"]
    assert len(non_root) == 2
    for n in non_root:
        assert n.is_valid is True
        assert n.metric == 0.01
