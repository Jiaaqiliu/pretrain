"""Statistical anomaly detection for training metrics.

Detects 6 types of anomalies using purely statistical methods (no LLM needed):
1. Loss spike — sudden jump in loss (z-score > threshold)
2. Gradient explosion — grad norm exceeds bounds
3. NaN/Inf — any metric is NaN or Inf
4. Throughput drop — significant decrease in tokens/second
5. Loss plateau — no improvement over N steps
6. Divergence — loss consistently increasing

Adapted from AutoTrain's monitor.py with improvements from OLMo-core's
StabilityMonitorCallback design (rolling window, checkpoint-able state).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from autopretrain.core.types import Anomaly, AnomalyType, MetricSnapshot


@dataclass
class DetectorConfig:
    """Configuration for anomaly detection thresholds."""

    # Loss spike
    loss_spike_sigma: float = 4.0
    loss_spike_min_history: int = 30

    # Gradient explosion
    grad_norm_sigma: float = 4.0
    grad_norm_absolute_max: float = 100.0
    grad_norm_min_history: int = 20

    # Throughput drop
    throughput_drop_fraction: float = 0.5
    throughput_min_history: int = 10

    # Loss plateau
    plateau_window: int = 500
    plateau_min_improvement: float = 0.001

    # Divergence
    divergence_window: int = 100
    divergence_increasing_fraction: float = 0.6


class AnomalyDetector:
    """Statistical anomaly detector for training metrics.

    Maintains internal state (rolling windows) for efficient computation.
    State can be serialized for checkpoint/resume.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def check(
        self,
        snapshot: MetricSnapshot,
        history: list[MetricSnapshot],
    ) -> list[Anomaly]:
        """Check a metric snapshot for anomalies given history.

        Returns list of detected anomalies (may be empty).
        """
        anomalies: list[Anomaly] = []

        # NaN/Inf detection (check first — invalidates other checks)
        nan_anomaly = self._check_nan(snapshot)
        if nan_anomaly:
            return [nan_anomaly]

        # Loss spike
        spike = self._check_loss_spike(snapshot, history)
        if spike:
            anomalies.append(spike)

        # Gradient explosion
        grad = self._check_gradient_explosion(snapshot, history)
        if grad:
            anomalies.append(grad)

        # Throughput drop
        tp = self._check_throughput_drop(snapshot, history)
        if tp:
            anomalies.append(tp)

        # Loss plateau
        plateau = self._check_plateau(snapshot, history)
        if plateau:
            anomalies.append(plateau)

        # Divergence
        div = self._check_divergence(snapshot, history)
        if div:
            anomalies.append(div)

        return anomalies

    def _check_nan(self, snapshot: MetricSnapshot) -> Anomaly | None:
        for key, value in snapshot.metrics.items():
            if math.isnan(value) or math.isinf(value):
                return Anomaly(
                    type=AnomalyType.NAN_DETECTED,
                    severity=1.0,
                    step=snapshot.step,
                    message=f"NaN/Inf in '{key}' at step {snapshot.step}",
                    evidence={"metric": key, "value": str(value)},
                )
        return None

    def _check_loss_spike(
        self, snapshot: MetricSnapshot, history: list[MetricSnapshot]
    ) -> Anomaly | None:
        loss = snapshot.loss
        if loss is None:
            return None

        losses = [s.loss for s in history if s.loss is not None]
        if len(losses) < self.config.loss_spike_min_history:
            return None

        mean = sum(losses) / len(losses)
        std = math.sqrt(sum((x - mean) ** 2 for x in losses) / len(losses))
        if std < 1e-10:
            return None

        z_score = (loss - mean) / std
        if z_score > self.config.loss_spike_sigma:
            severity = min(1.0, z_score / (2 * self.config.loss_spike_sigma))
            return Anomaly(
                type=AnomalyType.LOSS_SPIKE,
                severity=severity,
                step=snapshot.step,
                message=f"Loss spike: z={z_score:.1f} (loss={loss:.4f}, mean={mean:.4f})",
                evidence={"z_score": z_score, "loss": loss, "mean": mean, "std": std},
            )
        return None

    def _check_gradient_explosion(
        self, snapshot: MetricSnapshot, history: list[MetricSnapshot]
    ) -> Anomaly | None:
        grad = snapshot.grad_norm
        if grad is None:
            return None

        # Absolute threshold
        if grad > self.config.grad_norm_absolute_max:
            return Anomaly(
                type=AnomalyType.GRADIENT_EXPLOSION,
                severity=min(1.0, grad / (2 * self.config.grad_norm_absolute_max)),
                step=snapshot.step,
                message=f"Gradient explosion: norm={grad:.1f} > {self.config.grad_norm_absolute_max}",
                evidence={"grad_norm": grad},
            )

        # Z-score threshold
        grads = [s.grad_norm for s in history if s.grad_norm is not None]
        if len(grads) < self.config.grad_norm_min_history:
            return None

        mean = sum(grads) / len(grads)
        std = math.sqrt(sum((x - mean) ** 2 for x in grads) / len(grads))
        if std < 1e-10:
            return None

        z_score = (grad - mean) / std
        if z_score > self.config.grad_norm_sigma:
            severity = min(1.0, z_score / (2 * self.config.grad_norm_sigma))
            return Anomaly(
                type=AnomalyType.GRADIENT_EXPLOSION,
                severity=severity,
                step=snapshot.step,
                message=f"Gradient spike: z={z_score:.1f} (norm={grad:.2f}, mean={mean:.2f})",
                evidence={"z_score": z_score, "grad_norm": grad, "mean": mean},
            )
        return None

    def _check_throughput_drop(
        self, snapshot: MetricSnapshot, history: list[MetricSnapshot]
    ) -> Anomaly | None:
        tp = snapshot.throughput
        if tp is None:
            return None

        tps = [s.throughput for s in history if s.throughput is not None]
        if len(tps) < self.config.throughput_min_history:
            return None

        mean_tp = sum(tps) / len(tps)
        if mean_tp < 1e-6:
            return None

        drop = 1.0 - (tp / mean_tp)
        if drop > self.config.throughput_drop_fraction:
            return Anomaly(
                type=AnomalyType.THROUGHPUT_DROP,
                severity=min(1.0, drop),
                step=snapshot.step,
                message=f"Throughput dropped {drop*100:.0f}% (current={tp:.0f}, mean={mean_tp:.0f})",
                evidence={"drop_fraction": drop, "current": tp, "mean": mean_tp},
            )
        return None

    def _check_plateau(
        self, snapshot: MetricSnapshot, history: list[MetricSnapshot]
    ) -> Anomaly | None:
        loss = snapshot.loss
        if loss is None:
            return None
        if len(history) < self.config.plateau_window:
            return None

        recent = history[-self.config.plateau_window:]
        losses = [s.loss for s in recent if s.loss is not None]
        if len(losses) < self.config.plateau_window // 2:
            return None

        mid = len(losses) // 2
        first_mean = sum(losses[:mid]) / mid
        second_mean = sum(losses[mid:]) / (len(losses) - mid)

        if first_mean == 0:
            return None
        improvement = (first_mean - second_mean) / first_mean

        if abs(improvement) < self.config.plateau_min_improvement:
            return Anomaly(
                type=AnomalyType.LOSS_PLATEAU,
                severity=0.5,
                step=snapshot.step,
                message=f"Loss plateau: {improvement*100:.3f}% improvement over {len(losses)} steps",
                evidence={"improvement": improvement, "window": len(losses)},
            )
        return None

    def _check_divergence(
        self, snapshot: MetricSnapshot, history: list[MetricSnapshot]
    ) -> Anomaly | None:
        loss = snapshot.loss
        if loss is None:
            return None
        if len(history) < self.config.divergence_window:
            return None

        window = history[-self.config.divergence_window:]
        losses = [s.loss for s in window if s.loss is not None]
        n = len(losses)
        if n < self.config.divergence_window // 2:
            return None

        # Linear regression slope
        x_mean = (n - 1) / 2.0
        y_mean = sum(losses) / n
        numer = sum((i - x_mean) * (l - y_mean) for i, l in enumerate(losses))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        if denom == 0:
            return None

        slope = numer / denom
        normalized_slope = slope / y_mean if y_mean != 0 else slope

        if normalized_slope <= 0:
            return None

        # Check fraction of increasing steps
        increasing = sum(1 for i in range(1, n) if losses[i] > losses[i - 1])
        frac = increasing / (n - 1)

        if frac > self.config.divergence_increasing_fraction:
            severity = min(1.0, normalized_slope * 10)
            return Anomaly(
                type=AnomalyType.DIVERGENCE,
                severity=severity,
                step=snapshot.step,
                message=f"Divergence: slope={normalized_slope:.4f}, {frac*100:.0f}% increasing",
                evidence={"normalized_slope": normalized_slope, "increasing_fraction": frac},
            )
        return None
