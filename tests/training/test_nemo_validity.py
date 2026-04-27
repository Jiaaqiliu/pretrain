"""PR4 acceptance: check_validity enforces hard constraints."""

from __future__ import annotations

import math
from pathlib import Path

from agent_evolve.benchmarks.nemo_reasoner import NemoReasonerBenchmark
from agent_evolve.training.types import (
    CheckpointRef,
    EvalMetrics,
    TrainingTrialResult,
)
from agent_evolve.training.workspace import TrainingWorkspace


def _ok_metrics(value: float = 0.5) -> EvalMetrics:
    return EvalMetrics(
        primary_metric_name="local_holdout_pass_at_1", primary_metric_value=value
    )


def test_missing_checkpoint_is_invalid(minimal_workspace: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    trial = TrainingTrialResult(
        node_id="n1",
        workspace_path=str(ws.root),
        status="success",
        checkpoint=None,
        eval_metrics=_ok_metrics(),
    )
    report = NemoReasonerBenchmark().check_validity(ws, trial)
    assert not report.is_valid
    assert report.hard_fail_reason == "checkpoint_missing"


def test_nan_metric_is_invalid(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    trial = TrainingTrialResult(
        node_id="n2",
        workspace_path=str(ws.root),
        status="success",
        checkpoint=CheckpointRef(name="c", path=str(ckpt_dir)),
        eval_metrics=_ok_metrics(math.nan),
    )
    report = NemoReasonerBenchmark().check_validity(ws, trial)
    assert not report.is_valid
    assert report.hard_fail_reason == "metric_nan"


def test_missing_metrics_is_invalid(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    trial = TrainingTrialResult(
        node_id="n3",
        workspace_path=str(ws.root),
        status="success",
        checkpoint=CheckpointRef(name="c", path=str(ckpt_dir)),
        eval_metrics=None,
    )
    report = NemoReasonerBenchmark().check_validity(ws, trial)
    assert not report.is_valid
    assert report.hard_fail_reason == "metrics_missing"


def test_train_failed_is_invalid(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    trial = TrainingTrialResult(
        node_id="n4",
        workspace_path=str(ws.root),
        status="train_failed",
        checkpoint=CheckpointRef(name="c", path=str(ckpt_dir)),
    )
    report = NemoReasonerBenchmark().check_validity(ws, trial)
    assert not report.is_valid
    assert report.hard_fail_reason == "train_failed"


def test_valid_trial_passes(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    trial = TrainingTrialResult(
        node_id="n5",
        workspace_path=str(ws.root),
        status="success",
        checkpoint=CheckpointRef(name="c", path=str(ckpt_dir)),
        eval_metrics=_ok_metrics(),
    )
    report = NemoReasonerBenchmark().check_validity(ws, trial)
    assert report.is_valid
