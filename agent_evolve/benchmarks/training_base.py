"""Base protocol for training benchmarks.

Deliberately **not** a subclass of :class:`BenchmarkAdapter` because training
benchmarks evaluate checkpoints/adapters, not agent trajectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..training.types import (
    CheckpointRef,
    ErrorBuckets,
    EvalMetrics,
    EvalPlan,
    MetricSpec,
    TrainingTrialResult,
    ValidityReport,
)


@runtime_checkable
class TrainingBenchmarkAdapter(Protocol):
    name: str

    def primary_metric(self) -> MetricSpec: ...

    def build_eval_plan(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        split: str,
    ) -> EvalPlan: ...

    def evaluate(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        backend: Any,
        split: str,
    ) -> Any: ...

    def parse_metrics(self, result_dir: Path) -> EvalMetrics: ...

    def analyze_errors(
        self,
        result_dir: Path,
        metrics: EvalMetrics,
    ) -> ErrorBuckets: ...

    def check_validity(
        self,
        workspace: Any,
        trial_result: TrainingTrialResult,
    ) -> ValidityReport: ...


__all__ = ["TrainingBenchmarkAdapter"]
