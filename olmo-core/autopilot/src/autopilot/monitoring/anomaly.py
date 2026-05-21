"""Training anomaly detection.

Implements multi-signal anomaly detection for LLM training, inspired by:
- OLMo-core's SkipStepOptimizer (statistical threshold on loss)
- Wortsman et al. (2023) on activation/gradient norm monitoring
- PaLM's checkpoint rollback strategy for loss spikes
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from autopilot.monitoring.metrics import MetricsWindow
from autopilot.utils.logging import get_logger

log = get_logger("monitoring.anomaly")


class AnomalyType(enum.Enum):
    LOSS_SPIKE = "loss_spike"
    GRADIENT_EXPLOSION = "gradient_explosion"
    NAN_DETECTED = "nan_detected"
    SLOW_CONVERGENCE = "slow_convergence"
    THROUGHPUT_DROP = "throughput_drop"
    GPU_MEMORY_PRESSURE = "gpu_memory_pressure"
    DIVERGENCE = "divergence"


class Severity(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TrainingAnomaly:
    """A detected training anomaly with context."""

    anomaly_type: AnomalyType
    severity: Severity
    step: int
    value: float
    threshold: float
    message: str
    suggested_action: str
    context: dict = field(default_factory=dict)


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection thresholds."""

    loss_spike_sigma: float = 4.0
    loss_spike_critical_sigma: float = 8.0
    grad_norm_spike_sigma: float = 5.0
    throughput_drop_threshold: float = 0.5  # 50% drop triggers alert
    convergence_window: int = 500
    convergence_min_improvement: float = 0.001  # minimum loss improvement per window
    min_history_length: int = 50


class AnomalyDetector:
    """Detects training anomalies from metrics windows.

    Uses a combination of:
    1. Statistical outlier detection (sigma-based thresholds)
    2. Trend analysis (convergence rate)
    3. Absolute thresholds (NaN, throughput)
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self._config = config or AnomalyDetectorConfig()

    def detect(self, window: MetricsWindow) -> List[TrainingAnomaly]:
        """Run all anomaly detectors on the current metrics window."""
        anomalies: List[TrainingAnomaly] = []

        if window.length < self._config.min_history_length:
            return anomalies

        anomalies.extend(self._check_loss_spike(window))
        anomalies.extend(self._check_gradient_explosion(window))
        anomalies.extend(self._check_nan(window))
        anomalies.extend(self._check_convergence(window))
        anomalies.extend(self._check_throughput(window))

        return anomalies

    def _check_loss_spike(self, window: MetricsWindow) -> List[TrainingAnomaly]:
        anomalies = []
        latest = window.latest
        if latest is None or "loss" not in latest.metrics:
            return anomalies

        current_loss = latest.metrics["loss"]
        mean = window.mean("loss")
        std = window.std("loss")

        if mean is None or std is None or std < 1e-8:
            return anomalies

        deviation = (current_loss - mean) / std

        if deviation > self._config.loss_spike_critical_sigma:
            anomalies.append(
                TrainingAnomaly(
                    anomaly_type=AnomalyType.LOSS_SPIKE,
                    severity=Severity.CRITICAL,
                    step=latest.step,
                    value=current_loss,
                    threshold=mean + self._config.loss_spike_critical_sigma * std,
                    message=(
                        f"Critical loss spike: {current_loss:.4f} "
                        f"({deviation:.1f}σ above mean {mean:.4f})"
                    ),
                    suggested_action="rollback_checkpoint",
                    context={"deviation_sigma": deviation, "mean": mean, "std": std},
                )
            )
        elif deviation > self._config.loss_spike_sigma:
            anomalies.append(
                TrainingAnomaly(
                    anomaly_type=AnomalyType.LOSS_SPIKE,
                    severity=Severity.MEDIUM,
                    step=latest.step,
                    value=current_loss,
                    threshold=mean + self._config.loss_spike_sigma * std,
                    message=(
                        f"Loss spike detected: {current_loss:.4f} "
                        f"({deviation:.1f}σ above mean {mean:.4f})"
                    ),
                    suggested_action="skip_step",
                    context={"deviation_sigma": deviation, "mean": mean, "std": std},
                )
            )

        return anomalies

    def _check_gradient_explosion(self, window: MetricsWindow) -> List[TrainingAnomaly]:
        anomalies = []
        latest = window.latest
        if latest is None or "grad_norm" not in latest.metrics:
            return anomalies

        current_norm = latest.metrics["grad_norm"]
        mean = window.mean("grad_norm")
        std = window.std("grad_norm")

        if mean is None or std is None or std < 1e-8:
            return anomalies

        deviation = (current_norm - mean) / std

        if deviation > self._config.grad_norm_spike_sigma:
            severity = Severity.HIGH if deviation > 2 * self._config.grad_norm_spike_sigma else Severity.MEDIUM
            anomalies.append(
                TrainingAnomaly(
                    anomaly_type=AnomalyType.GRADIENT_EXPLOSION,
                    severity=severity,
                    step=latest.step,
                    value=current_norm,
                    threshold=mean + self._config.grad_norm_spike_sigma * std,
                    message=(
                        f"Gradient norm spike: {current_norm:.2f} "
                        f"({deviation:.1f}σ above mean {mean:.2f})"
                    ),
                    suggested_action="reduce_lr" if severity == Severity.MEDIUM else "rollback_checkpoint",
                    context={"deviation_sigma": deviation, "mean": mean, "std": std},
                )
            )

        return anomalies

    def _check_nan(self, window: MetricsWindow) -> List[TrainingAnomaly]:
        anomalies = []
        latest = window.latest
        if latest is None:
            return anomalies

        for name, value in latest.metrics.items():
            if np.isnan(value) or np.isinf(value):
                anomalies.append(
                    TrainingAnomaly(
                        anomaly_type=AnomalyType.NAN_DETECTED,
                        severity=Severity.CRITICAL,
                        step=latest.step,
                        value=float("nan"),
                        threshold=0.0,
                        message=f"NaN/Inf detected in metric '{name}' at step {latest.step}",
                        suggested_action="rollback_checkpoint",
                        context={"metric_name": name},
                    )
                )
                break  # one NaN alert is enough

        return anomalies

    def _check_convergence(self, window: MetricsWindow) -> List[TrainingAnomaly]:
        anomalies = []
        if window.length < self._config.convergence_window:
            return anomalies

        loss_values = window.get_values("loss")
        if len(loss_values) < self._config.convergence_window:
            return anomalies

        # Compare first half vs second half of the window
        half = len(loss_values) // 2
        first_half_mean = np.mean(loss_values[:half])
        second_half_mean = np.mean(loss_values[half:])

        improvement = (first_half_mean - second_half_mean) / abs(first_half_mean)

        if improvement < self._config.convergence_min_improvement:
            latest = window.latest
            anomalies.append(
                TrainingAnomaly(
                    anomaly_type=AnomalyType.SLOW_CONVERGENCE,
                    severity=Severity.LOW if improvement > 0 else Severity.MEDIUM,
                    step=latest.step if latest else 0,
                    value=improvement,
                    threshold=self._config.convergence_min_improvement,
                    message=(
                        f"Slow convergence: {improvement*100:.3f}% improvement "
                        f"over last {self._config.convergence_window} steps "
                        f"(threshold: {self._config.convergence_min_improvement*100:.3f}%)"
                    ),
                    suggested_action="consider_early_stop" if improvement <= 0 else "adjust_lr",
                    context={
                        "improvement_pct": improvement * 100,
                        "first_half_mean": float(first_half_mean),
                        "second_half_mean": float(second_half_mean),
                    },
                )
            )

        return anomalies

    def _check_throughput(self, window: MetricsWindow) -> List[TrainingAnomaly]:
        anomalies = []
        latest = window.latest
        if latest is None or "throughput" not in latest.metrics:
            return anomalies

        current_throughput = latest.metrics["throughput"]
        mean = window.mean("throughput")

        if mean is None or mean < 1e-8:
            return anomalies

        ratio = current_throughput / mean

        if ratio < self._config.throughput_drop_threshold:
            anomalies.append(
                TrainingAnomaly(
                    anomaly_type=AnomalyType.THROUGHPUT_DROP,
                    severity=Severity.MEDIUM,
                    step=latest.step,
                    value=current_throughput,
                    threshold=mean * self._config.throughput_drop_threshold,
                    message=(
                        f"Throughput drop: {current_throughput:.0f} tokens/s "
                        f"({ratio*100:.0f}% of average {mean:.0f})"
                    ),
                    suggested_action="check_hardware",
                    context={"ratio": ratio, "mean_throughput": mean},
                )
            )

        return anomalies
