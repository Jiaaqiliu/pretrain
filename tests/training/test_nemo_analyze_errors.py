"""PR4 acceptance: analyze_errors produces nonzero buckets for format errors."""

from __future__ import annotations

import json
from pathlib import Path

from agent_evolve.benchmarks.nemo_reasoner import (
    DEFAULT_PRIMARY_METRIC_NAME,
    NemoReasonerBenchmark,
)
from agent_evolve.training.types import EvalMetrics


def _write_predictions(dir_: Path, rows: list[dict]) -> None:
    with open(dir_ / "predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_format_error_bucket_nonzero(tmp_path: Path) -> None:
    _write_predictions(
        tmp_path,
        [
            {"is_correct": False, "format_error": True},
            {"is_correct": False, "format_error": True},
            {"is_correct": True},
        ],
    )
    bench = NemoReasonerBenchmark()
    metrics = EvalMetrics(
        primary_metric_name=DEFAULT_PRIMARY_METRIC_NAME, primary_metric_value=0.33
    )
    buckets = bench.analyze_errors(tmp_path, metrics)
    assert buckets.counts.get("format_error", 0) == 2


def test_empty_predictions_returns_empty_buckets(tmp_path: Path) -> None:
    (tmp_path / "predictions.jsonl").write_text("")
    bench = NemoReasonerBenchmark()
    buckets = bench.analyze_errors(
        tmp_path, EvalMetrics(primary_metric_name="x", primary_metric_value=0.0)
    )
    assert buckets.counts == {}
