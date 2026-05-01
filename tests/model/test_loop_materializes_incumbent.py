"""PR3 acceptance: loop materializes incumbent only when incumbent_changed=True."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_evolve.model.loop import TrainingEvolutionLoop
from agent_evolve.model.types import (
    CheckpointRef,
    MCGSCycleReport,
    TrainingSearchNode,
)
from agent_evolve.model.workspace import TrainingWorkspace

from .fakes import FakeBackend, FakeBenchmark


@dataclass
class _FakeGraph:
    nodes: dict[str, TrainingSearchNode] = field(default_factory=dict)


class _StubAlgoNoChange:
    def __init__(self) -> None:
        self.graph = _FakeGraph()
        self.cycle = 0

    def run_cycle(self, ctx):
        self.cycle += 1
        return MCGSCycleReport(
            cycle=self.cycle,
            selected_parent_id=None,
            trial_node_ids=[],
            incumbent_node_id=None,
            incumbent_changed=False,
            best_metric=None,
            graph_path="",
            report_path="",
        )


class _StubAlgoChange:
    def __init__(self) -> None:
        node = TrainingSearchNode(
            node_id="incumbent-1",
            parent_id=None,
            branch_id=0,
            metric=0.42,
            reward=0.1,
            checkpoint=CheckpointRef(name="ckpt", path="/tmp/ckpt"),
        )
        self.graph = _FakeGraph(nodes={node.node_id: node})
        self.cycle = 0

    def run_cycle(self, ctx):
        self.cycle += 1
        return MCGSCycleReport(
            cycle=self.cycle,
            selected_parent_id=None,
            trial_node_ids=["incumbent-1"],
            incumbent_node_id="incumbent-1",
            incumbent_changed=True,
            best_metric=0.42,
            graph_path="",
            report_path="",
        )


def test_no_materialization_when_unchanged(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=_StubAlgoNoChange(),
        backend=FakeBackend(),
        work_dir=tmp_path / "work",
    )
    loop.run(cycles=1)
    assert not (minimal_workspace / "evolution" / "incumbent.json").exists()


def test_materialize_when_incumbent_changed(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    algo = _StubAlgoChange()
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=algo,
        backend=FakeBackend(),
        work_dir=tmp_path / "work",
    )
    result = loop.run(cycles=1)
    assert (minimal_workspace / "evolution" / "incumbent.json").exists()
    assert result.best_metric == 0.42
    assert result.incumbent_node_id == "incumbent-1"
