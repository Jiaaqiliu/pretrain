"""Lightweight fakes used across training tests.

Intentionally live in the test tree so they don't pollute the shipped package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_evolve.model.types import (
    CheckpointRef,
    EvalMetrics,
    ErrorBuckets,
    MetricSpec,
    TrainingTrialResult,
    ValidityReport,
)


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_trial(self, workspace, node, budget, benchmark) -> TrainingTrialResult:
        self.calls.append({"node": node.node_id})
        ckpt = CheckpointRef(name=node.node_id, path=str(Path(workspace.root) / "ckpt"))
        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=str(workspace.root),
            status="success",
            checkpoint=ckpt,
            eval_metrics=EvalMetrics(
                primary_metric_name="fake_metric", primary_metric_value=0.5
            ),
            error_buckets=ErrorBuckets(counts={}),
            validity=ValidityReport(is_valid=True),
        )


class FakeBenchmark:
    name = "fake_benchmark"

    def primary_metric(self) -> MetricSpec:
        return MetricSpec(name="fake_metric", maximize=True)

    def parse_metrics(self, result_dir: Path) -> EvalMetrics:
        return EvalMetrics(primary_metric_name="fake_metric", primary_metric_value=0.0)

    def analyze_errors(self, result_dir: Path, metrics) -> ErrorBuckets:
        return ErrorBuckets(counts={})

    def check_validity(self, workspace, trial_result) -> ValidityReport:
        return ValidityReport(is_valid=True)

    def build_eval_plan(self, workspace, checkpoint, split):  # noqa: ARG002
        return None

    def evaluate(self, workspace, checkpoint, backend, split):  # noqa: ARG002
        return None


class FakeAlgorithm:
    def __init__(self) -> None:
        self.run_cycle_calls = 0

    def run_cycle(self, ctx):
        from agent_evolve.model.types import MCGSCycleReport

        self.run_cycle_calls += 1
        return MCGSCycleReport(
            cycle=self.run_cycle_calls,
            selected_parent_id=None,
            trial_node_ids=[],
            incumbent_node_id=None,
            incumbent_changed=False,
            best_metric=None,
            graph_path="",
            report_path="",
        )
