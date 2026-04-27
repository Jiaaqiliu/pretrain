"""PR2 acceptance: mock single-node backend returns TrainingTrialResult."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend
from agent_evolve.training.types import (
    TrainingSearchNode,
    TrainingTrialResult,
    TrialBudget,
)
from agent_evolve.training.workspace import TrainingWorkspace

from .fakes import FakeBenchmark


def _make_node() -> TrainingSearchNode:
    return TrainingSearchNode(node_id="node-mock", parent_id=None, branch_id=0)


def test_run_trial_returns_result(minimal_workspace: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    backend = SingleNodeTinkerLiteBackend(mock=True)
    result = backend.run_trial(ws, _make_node(), TrialBudget(seconds=30), FakeBenchmark())
    assert isinstance(result, TrainingTrialResult)
    assert result.status == "success"
    assert result.checkpoint is not None


def test_train_failed_when_pipeline_missing(minimal_workspace: Path) -> None:
    (minimal_workspace / "train" / "pipeline.yaml").unlink()
    ws = TrainingWorkspace(minimal_workspace)  # skip validation
    backend = SingleNodeTinkerLiteBackend(mock=True)
    result = backend.run_trial(ws, _make_node(), TrialBudget(seconds=30), FakeBenchmark())
    assert result.status == "train_failed"


def test_candidate_workspace_not_deleted(minimal_workspace: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    backend = SingleNodeTinkerLiteBackend(mock=True)
    backend.run_trial(ws, _make_node(), TrialBudget(seconds=30), FakeBenchmark())
    assert minimal_workspace.is_dir()
    assert (minimal_workspace / "train" / "pipeline.yaml").exists()


def test_eval_failed_status_on_benchmark_exception(minimal_workspace: Path) -> None:
    class FaultyBenchmark(FakeBenchmark):
        def parse_metrics(self, result_dir):
            raise RuntimeError("parse failed")

    # We can't trigger eval_failed through `parse_metrics` alone because the
    # backend guards it with try/except; instead break ``primary_metric`` which
    # is called upstream without a guard.
    class HarderFaultyBenchmark(FakeBenchmark):
        def primary_metric(self):
            raise RuntimeError("primary_metric explodes")

    ws = TrainingWorkspace.load(minimal_workspace)
    backend = SingleNodeTinkerLiteBackend(mock=True)
    result = backend.run_trial(
        ws, _make_node(), TrialBudget(seconds=30), HarderFaultyBenchmark()
    )
    assert result.status == "eval_failed"
