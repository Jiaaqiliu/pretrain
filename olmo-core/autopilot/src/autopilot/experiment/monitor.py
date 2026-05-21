"""Experiment monitoring — continuous observation of running experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from autopilot.backends.base import ComputeBackend, JobHandle, JobStatus
from autopilot.monitoring.anomaly import AnomalyDetector, AnomalyDetectorConfig, TrainingAnomaly
from autopilot.monitoring.metrics import MetricsCollector, MetricsSnapshot
from autopilot.monitoring.prediction import LossPredictor, TrainingOutlook
from autopilot.optimization.early_stopping import EarlyStoppingStrategy, StoppingDecision
from autopilot.utils.logging import get_logger

log = get_logger("experiment.monitor")


@dataclass
class ExperimentState:
    """Current state of a monitored experiment."""

    experiment_id: str
    handle: JobHandle
    status: str = "running"
    last_metrics: Optional[MetricsSnapshot] = None
    anomalies: List[TrainingAnomaly] = field(default_factory=list)
    outlook: Optional[TrainingOutlook] = None
    stopping_decision: Optional[StoppingDecision] = None


class ExperimentMonitor:
    """Monitors multiple running experiments and detects issues.

    Orchestrates:
    - Metrics collection from all experiments
    - Anomaly detection (loss spikes, gradient explosions, etc.)
    - Performance prediction (final loss extrapolation)
    - Early stopping decisions
    - Cross-experiment comparison
    """

    def __init__(
        self,
        backend: ComputeBackend,
        total_steps: int = 100000,
        target_loss: Optional[float] = None,
        anomaly_config: Optional[AnomalyDetectorConfig] = None,
        on_anomaly: Optional[Callable[[str, TrainingAnomaly], None]] = None,
        on_stop_decision: Optional[Callable[[str, StoppingDecision], None]] = None,
    ):
        self._backend = backend
        self._metrics = MetricsCollector()
        self._anomaly_detector = AnomalyDetector(anomaly_config)
        self._early_stopping = EarlyStoppingStrategy(
            total_steps=total_steps, target_loss=target_loss
        )
        self._predictor = LossPredictor(total_steps=total_steps, target_loss=target_loss)
        self._experiments: Dict[str, ExperimentState] = {}
        self._on_anomaly = on_anomaly
        self._on_stop_decision = on_stop_decision

    def add_experiment(self, experiment_id: str, handle: JobHandle) -> None:
        """Start monitoring an experiment."""
        self._experiments[experiment_id] = ExperimentState(
            experiment_id=experiment_id, handle=handle
        )
        log.info(f"Now monitoring experiment {experiment_id}")

    def remove_experiment(self, experiment_id: str) -> None:
        """Stop monitoring an experiment."""
        self._experiments.pop(experiment_id, None)

    @property
    def monitored_experiments(self) -> List[str]:
        return list(self._experiments.keys())

    def poll(self) -> Dict[str, ExperimentState]:
        """Poll all monitored experiments for updates.

        This is the main monitoring loop step. Call periodically.
        Returns updated states for all experiments.
        """
        for eid, state in list(self._experiments.items()):
            try:
                self._poll_experiment(eid, state)
            except Exception as e:
                log.warning(f"Error polling experiment {eid}: {e}")

        return dict(self._experiments)

    def get_state(self, experiment_id: str) -> Optional[ExperimentState]:
        return self._experiments.get(experiment_id)

    def get_comparison(self, metric: str = "loss") -> Dict[str, Dict]:
        """Compare all active experiments on a given metric."""
        experiment_ids = list(self._experiments.keys())
        return self._metrics.compare_experiments(experiment_ids, metric)

    def get_rankings(self, metric: str = "loss") -> List[tuple]:
        """Rank experiments by current metric value (ascending)."""
        comparison = self.get_comparison(metric)
        ranked = []
        for eid, data in comparison.items():
            current = data.get("current")
            if current is not None:
                ranked.append((eid, current))
        ranked.sort(key=lambda x: x[1])
        return ranked

    def _poll_experiment(self, experiment_id: str, state: ExperimentState) -> None:
        """Poll a single experiment for updates."""
        # Check job status
        job_status = self._backend.get_status(state.handle)
        if job_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            state.status = job_status.value
            return

        # Get latest metrics
        metrics = self._backend.get_metrics(state.handle)
        if metrics is None:
            return

        snapshot = MetricsSnapshot.from_job_metrics(metrics)
        self._metrics.record(experiment_id, snapshot)
        state.last_metrics = snapshot

        # Run anomaly detection
        window = self._metrics.get_window(experiment_id)
        if window:
            anomalies = self._anomaly_detector.detect(window)
            if anomalies:
                state.anomalies = anomalies
                for anomaly in anomalies:
                    log.warning(
                        f"[{experiment_id}] Anomaly: {anomaly.message} "
                        f"(severity={anomaly.severity.value})"
                    )
                    if self._on_anomaly:
                        self._on_anomaly(experiment_id, anomaly)

            # Run prediction
            outlook = self._predictor.predict(window)
            if outlook:
                state.outlook = outlook

            # Check early stopping
            decision = self._early_stopping.should_stop(experiment_id, window)
            if decision.should_stop:
                state.stopping_decision = decision
                log.info(
                    f"[{experiment_id}] Early stop recommended: {decision.message} "
                    f"(confidence={decision.confidence:.2f})"
                )
                if self._on_stop_decision:
                    self._on_stop_decision(experiment_id, decision)

    def run_asha_evaluation(self) -> List[tuple]:
        """Run ASHA evaluation across all experiments at appropriate rungs."""
        rungs = self._early_stopping.get_asha_rungs()
        windows = {}
        for eid in self._experiments:
            w = self._metrics.get_window(eid)
            if w:
                windows[eid] = w

        all_decisions = []
        for rung in rungs:
            decisions = self._early_stopping.asha_evaluate(windows, rung)
            all_decisions.extend(decisions)

        return all_decisions
