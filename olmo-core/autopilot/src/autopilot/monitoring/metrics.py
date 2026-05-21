"""Training metrics collection and windowed statistics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from autopilot.backends.base import JobMetrics
from autopilot.utils.logging import get_logger

log = get_logger("monitoring.metrics")


@dataclass
class MetricsSnapshot:
    """A timestamped collection of training metrics."""

    timestamp: float
    step: int
    metrics: Dict[str, float]

    @classmethod
    def from_job_metrics(cls, jm: JobMetrics) -> "MetricsSnapshot":
        m: Dict[str, float] = {}
        if jm.loss is not None:
            m["loss"] = jm.loss
        if jm.learning_rate is not None:
            m["learning_rate"] = jm.learning_rate
        if jm.grad_norm is not None:
            m["grad_norm"] = jm.grad_norm
        if jm.throughput_tokens_per_sec is not None:
            m["throughput"] = jm.throughput_tokens_per_sec
        if jm.gpu_utilization is not None:
            m["gpu_utilization"] = jm.gpu_utilization
        if jm.gpu_memory_used_gb is not None:
            m["gpu_memory_gb"] = jm.gpu_memory_used_gb
        m.update(jm.custom)
        return cls(timestamp=time.time(), step=jm.step, metrics=m)


@dataclass
class MetricsWindow:
    """Rolling window of metrics with statistics."""

    window_size: int = 128
    _history: Deque[MetricsSnapshot] = field(default_factory=lambda: deque(maxlen=128))

    def __post_init__(self):
        self._history = deque(maxlen=self.window_size)

    def add(self, snapshot: MetricsSnapshot) -> None:
        self._history.append(snapshot)

    @property
    def length(self) -> int:
        return len(self._history)

    @property
    def latest(self) -> Optional[MetricsSnapshot]:
        return self._history[-1] if self._history else None

    def get_values(self, metric_name: str) -> np.ndarray:
        values = [s.metrics.get(metric_name) for s in self._history if metric_name in s.metrics]
        return np.array(values, dtype=np.float64)

    def mean(self, metric_name: str) -> Optional[float]:
        values = self.get_values(metric_name)
        return float(np.mean(values)) if len(values) > 0 else None

    def std(self, metric_name: str) -> Optional[float]:
        values = self.get_values(metric_name)
        return float(np.std(values)) if len(values) > 1 else None

    def trend(self, metric_name: str, last_n: int = 20) -> Optional[float]:
        """Compute linear trend (slope) of a metric over the last N steps."""
        values = self.get_values(metric_name)
        if len(values) < max(5, last_n // 2):
            return None
        values = values[-last_n:]
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)

    def rate_of_change(self, metric_name: str) -> Optional[float]:
        """Relative rate of change: (latest - earliest_in_window) / earliest."""
        values = self.get_values(metric_name)
        if len(values) < 2:
            return None
        first, last = values[0], values[-1]
        if abs(first) < 1e-10:
            return None
        return float((last - first) / abs(first))


class MetricsCollector:
    """Collects and maintains metrics history for multiple experiments."""

    def __init__(self, window_size: int = 256):
        self._windows: Dict[str, MetricsWindow] = {}
        self._window_size = window_size
        self._full_history: Dict[str, List[MetricsSnapshot]] = {}

    def record(self, experiment_id: str, snapshot: MetricsSnapshot) -> None:
        if experiment_id not in self._windows:
            self._windows[experiment_id] = MetricsWindow(window_size=self._window_size)
            self._full_history[experiment_id] = []
        self._windows[experiment_id].add(snapshot)
        self._full_history[experiment_id].append(snapshot)

    def get_window(self, experiment_id: str) -> Optional[MetricsWindow]:
        return self._windows.get(experiment_id)

    def get_full_history(self, experiment_id: str) -> List[MetricsSnapshot]:
        return self._full_history.get(experiment_id, [])

    def get_latest(self, experiment_id: str) -> Optional[MetricsSnapshot]:
        window = self._windows.get(experiment_id)
        return window.latest if window else None

    def compare_experiments(
        self, experiment_ids: List[str], metric_name: str = "loss"
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Compare metrics across experiments."""
        results = {}
        for eid in experiment_ids:
            window = self._windows.get(eid)
            if window:
                results[eid] = {
                    "current": window.latest.metrics.get(metric_name) if window.latest else None,
                    "mean": window.mean(metric_name),
                    "trend": window.trend(metric_name),
                    "step": window.latest.step if window.latest else None,
                }
        return results
