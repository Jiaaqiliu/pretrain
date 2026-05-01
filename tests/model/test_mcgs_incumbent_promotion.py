"""PR5 acceptance: higher-scoring valid trial becomes incumbent."""

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


class _RisingBackend:
    """Returns monotonically increasing primary metric per call."""

    name = "rising"

    def __init__(self) -> None:
        self._i = 0

    def run_trial(self, workspace, node, budget, benchmark):
        self._i += 1
        ckpt_dir = Path(workspace.root) / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=str(workspace.root),
            status="success",
            checkpoint=CheckpointRef(name=f"c{self._i}", path=str(ckpt_dir)),
            eval_metrics=EvalMetrics(
                primary_metric_name="fake_metric",
                primary_metric_value=0.1 * self._i,
            ),
            error_buckets=ErrorBuckets(counts={}),
            validity=ValidityReport(is_valid=True),
        )


def test_higher_score_becomes_incumbent(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    algo = MCGSSearch()
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=algo,
        backend=_RisingBackend(),
        work_dir=tmp_path / "work",
    )
    result = loop.run(cycles=3)
    # incumbent should point to the third (highest) node.
    non_root = [n for n in algo.graph.nodes.values() if n.node_id != "node-root"]
    best = max(non_root, key=lambda n: n.metric or 0)
    assert algo.promotion_policy.incumbent_id == best.node_id
    assert result.best_metric is not None and result.best_metric >= 0.3
