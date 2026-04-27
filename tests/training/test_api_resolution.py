"""PR1 acceptance: TrainingEvolver resolves workspace/benchmark/algorithm/backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_evolve.training.api import TrainingEvolver
from agent_evolve.training.types import (
    TrainingRegistryError,
    TrainingWorkspaceNotFound,
)

from .fakes import FakeAlgorithm, FakeBackend, FakeBenchmark


def test_resolve_mock_components(minimal_workspace: Path, tmp_path: Path) -> None:
    evolver = TrainingEvolver(
        workspace=minimal_workspace,
        benchmark=FakeBenchmark(),
        algorithm=FakeAlgorithm(),
        backend=FakeBackend(),
        work_dir=tmp_path / "work",
    )
    assert isinstance(evolver.benchmark, FakeBenchmark)
    assert isinstance(evolver.algorithm, FakeAlgorithm)
    assert isinstance(evolver.backend, FakeBackend)


def test_missing_workspace_raises(tmp_path: Path) -> None:
    with pytest.raises(TrainingWorkspaceNotFound):
        TrainingEvolver(
            workspace=tmp_path / "nope",
            benchmark=FakeBenchmark(),
            algorithm=FakeAlgorithm(),
            backend=FakeBackend(),
            work_dir=tmp_path / "work",
        )


def test_unknown_benchmark_raises(minimal_workspace: Path, tmp_path: Path) -> None:
    with pytest.raises(TrainingRegistryError):
        TrainingEvolver(
            workspace=minimal_workspace,
            benchmark="not_a_real_benchmark",
            algorithm=FakeAlgorithm(),
            backend=FakeBackend(),
            work_dir=tmp_path / "work",
        )


def test_unknown_algorithm_raises(minimal_workspace: Path, tmp_path: Path) -> None:
    with pytest.raises(TrainingRegistryError):
        TrainingEvolver(
            workspace=minimal_workspace,
            benchmark=FakeBenchmark(),
            algorithm="not_a_real_algorithm",
            backend=FakeBackend(),
            work_dir=tmp_path / "work",
        )


def test_unknown_backend_raises(minimal_workspace: Path, tmp_path: Path) -> None:
    with pytest.raises(TrainingRegistryError):
        TrainingEvolver(
            workspace=minimal_workspace,
            benchmark=FakeBenchmark(),
            algorithm=FakeAlgorithm(),
            backend="not_a_real_backend",
            work_dir=tmp_path / "work",
        )
