"""PR3 acceptance: loop calls algorithm.run_cycle once per cycle."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.model.algorithms import NullSearchAlgorithm
from agent_evolve.model.loop import TrainingEvolutionLoop
from agent_evolve.model.workspace import TrainingWorkspace

from .fakes import FakeBackend, FakeBenchmark


def test_run_cycle_called_once(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    algo = NullSearchAlgorithm()
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=algo,
        backend=FakeBackend(),
        work_dir=tmp_path / "work",
    )
    result = loop.run(cycles=1)
    assert algo.cycle == 1
    assert result.cycles_completed == 1


def test_run_cycle_called_three_times(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    algo = NullSearchAlgorithm()
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=algo,
        backend=FakeBackend(),
        work_dir=tmp_path / "work",
    )
    loop.run(cycles=3)
    assert algo.cycle == 3
