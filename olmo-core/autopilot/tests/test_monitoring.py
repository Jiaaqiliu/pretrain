"""Tests for the monitoring module."""

import numpy as np

from autopilot.monitoring.anomaly import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyType,
    Severity,
)
from autopilot.monitoring.metrics import MetricsCollector, MetricsSnapshot, MetricsWindow
from autopilot.monitoring.prediction import LossPredictor


class TestMetricsWindow:
    def test_basic_statistics(self):
        window = MetricsWindow(window_size=100)
        for i in range(50):
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": 3.0 - i * 0.01}))

        assert window.length == 50
        assert window.mean("loss") is not None
        assert abs(window.mean("loss") - 2.755) < 0.01

    def test_trend_detection(self):
        window = MetricsWindow(window_size=100)
        # Decreasing loss
        for i in range(50):
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": 3.0 - i * 0.02}))

        trend = window.trend("loss")
        assert trend is not None
        assert trend < 0  # loss should be decreasing

    def test_increasing_trend(self):
        window = MetricsWindow(window_size=100)
        # Increasing loss (diverging)
        for i in range(50):
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": 2.0 + i * 0.01}))

        trend = window.trend("loss")
        assert trend is not None
        assert trend > 0  # loss increasing


class TestAnomalyDetector:
    def _make_window_with_spike(self, spike_at: int = 49, spike_value: float = 10.0):
        window = MetricsWindow(window_size=128)
        for i in range(50):
            loss = 2.5 + np.random.normal(0, 0.05) if i != spike_at else spike_value
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": loss}))
        return window

    def test_detects_loss_spike(self):
        window = self._make_window_with_spike(spike_value=10.0)
        detector = AnomalyDetector(AnomalyDetectorConfig(loss_spike_sigma=3.0))

        anomalies = detector.detect(window)
        loss_spikes = [a for a in anomalies if a.anomaly_type == AnomalyType.LOSS_SPIKE]
        assert len(loss_spikes) > 0
        assert loss_spikes[0].severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)

    def test_no_false_positive_on_normal_training(self):
        window = MetricsWindow(window_size=128)
        for i in range(100):
            loss = 3.0 - i * 0.01 + np.random.normal(0, 0.02)
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": loss}))

        detector = AnomalyDetector()
        anomalies = detector.detect(window)
        critical = [a for a in anomalies if a.severity == Severity.CRITICAL]
        assert len(critical) == 0

    def test_detects_nan(self):
        window = MetricsWindow(window_size=128)
        for i in range(60):
            loss = 2.5 if i != 59 else float("nan")
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": loss}))

        detector = AnomalyDetector()
        anomalies = detector.detect(window)
        nan_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.NAN_DETECTED]
        assert len(nan_anomalies) > 0

    def test_detects_slow_convergence(self):
        config = AnomalyDetectorConfig(convergence_window=50, convergence_min_improvement=0.01)
        detector = AnomalyDetector(config)

        window = MetricsWindow(window_size=256)
        # Flat loss (no convergence)
        for i in range(100):
            window.add(
                MetricsSnapshot(
                    timestamp=float(i), step=i, metrics={"loss": 2.5 + np.random.normal(0, 0.001)}
                )
            )

        anomalies = detector.detect(window)
        slow = [a for a in anomalies if a.anomaly_type == AnomalyType.SLOW_CONVERGENCE]
        assert len(slow) > 0


class TestLossPredictor:
    def test_prediction_on_decreasing_loss(self):
        predictor = LossPredictor(total_steps=10000)
        window = MetricsWindow(window_size=256)

        # Simulate power-law decay
        for i in range(1, 201):
            loss = 1.5 + 3.0 * (i ** -0.5)
            window.add(MetricsSnapshot(timestamp=float(i), step=i * 10, metrics={"loss": loss}))

        outlook = predictor.predict(window)
        assert outlook is not None
        assert outlook.predicted_final_loss < window.latest.metrics["loss"]
        assert 0 < outlook.confidence <= 1.0


class TestMetricsCollector:
    def test_record_and_retrieve(self):
        collector = MetricsCollector()
        for i in range(10):
            snapshot = MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": 3.0 - i * 0.1})
            collector.record("exp_1", snapshot)

        window = collector.get_window("exp_1")
        assert window is not None
        assert window.length == 10

    def test_compare_experiments(self):
        collector = MetricsCollector()
        for i in range(20):
            collector.record(
                "exp_1", MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": 3.0 - i * 0.05})
            )
            collector.record(
                "exp_2", MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": 3.5 - i * 0.03})
            )

        comparison = collector.compare_experiments(["exp_1", "exp_2"], "loss")
        assert "exp_1" in comparison
        assert "exp_2" in comparison
        assert comparison["exp_1"]["current"] < comparison["exp_2"]["current"]
