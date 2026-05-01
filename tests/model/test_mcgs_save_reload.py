"""PR5 acceptance: graph survives a save/reload round-trip."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.model.algorithms.mcgs import MCGSSearch
from agent_evolve.model.algorithms.mcgs.graph import TrainingSearchGraph
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


class _StaticBackend:
    name = "static"

    def run_trial(self, workspace, node, budget, benchmark):
        ckpt_dir = Path(workspace.root) / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=str(workspace.root),
            status="success",
            checkpoint=CheckpointRef(name="c", path=str(ckpt_dir)),
            eval_metrics=EvalMetrics(
                primary_metric_name="fake_metric", primary_metric_value=0.42
            ),
            error_buckets=ErrorBuckets(counts={"format_error": 1}),
            validity=ValidityReport(is_valid=True),
        )


def test_graph_roundtrip(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    algo = MCGSSearch()
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=algo,
        backend=_StaticBackend(),
        work_dir=tmp_path / "work",
    )
    loop.run(cycles=2)
    graph_path = Path(algo.graph_path)
    assert graph_path.exists()

    reloaded = TrainingSearchGraph.load(graph_path)
    assert len(reloaded) == len(algo.graph)
    for node_id, node in algo.graph.nodes.items():
        copy = reloaded.nodes[node_id]
        assert copy.parent_id == node.parent_id
        assert copy.branch_id == node.branch_id
        assert copy.visits == node.visits
        assert (copy.metric or 0) == (node.metric or 0)
        assert (copy.reward or 0) == (node.reward or 0)
