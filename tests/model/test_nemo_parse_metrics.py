"""PR4 acceptance: parse_metrics reads metrics.json into EvalMetrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.benchmarks.nemo_reasoner import (
    DEFAULT_PRIMARY_METRIC_NAME,
    NemoReasonerBenchmark,
)


def test_reads_primary_metric(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps({DEFAULT_PRIMARY_METRIC_NAME: 0.681, "format_error_rate": 0.02})
    )
    bench = NemoReasonerBenchmark()
    metrics = bench.parse_metrics(tmp_path)
    assert metrics.primary_metric_name == DEFAULT_PRIMARY_METRIC_NAME
    assert metrics.primary_metric_value == 0.681
    assert metrics.secondary.get("format_error_rate") == 0.02


def test_missing_metrics_file_raises(tmp_path: Path) -> None:
    bench = NemoReasonerBenchmark()
    with pytest.raises(FileNotFoundError):
        bench.parse_metrics(tmp_path)


def test_missing_primary_key_raises(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(json.dumps({"format_error_rate": 0.02}))
    bench = NemoReasonerBenchmark()
    with pytest.raises(KeyError):
        bench.parse_metrics(tmp_path)
