"""PR3 acceptance: each cycle writes a JSON report."""

from __future__ import annotations

import json
from pathlib import Path

from agent_evolve.model.algorithms import NullSearchAlgorithm
from agent_evolve.model.loop import TrainingEvolutionLoop
from agent_evolve.model.workspace import TrainingWorkspace

from .fakes import FakeBackend, FakeBenchmark


def test_cycle_report_written(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=NullSearchAlgorithm(),
        backend=FakeBackend(),
        work_dir=tmp_path / "work",
    )
    loop.run(cycles=2)
    reports_dir = minimal_workspace / "evolution" / "reports"
    assert reports_dir.is_dir()
    reports = sorted(reports_dir.glob("cycle_*.json"))
    assert len(reports) == 2
    with open(reports[0]) as f:
        data = json.load(f)
    assert data["cycle"] == 1
