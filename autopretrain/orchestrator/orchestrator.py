"""Main orchestrator: the autonomous training agent event loop.

This is the central brain that:
1. Submits training trials to the cluster
2. Monitors them in real-time via heartbeats and metrics
3. Detects failures and anomalies
4. Diagnoses root causes
5. Selects and executes recovery actions
6. Learns from outcomes for future improvement
7. Manages budget and scheduling
8. Reports results

The orchestrator runs as a long-lived async process on a CPU pod,
managing GPU training jobs without human intervention.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopretrain.core.types import (
    ActionType,
    Anomaly,
    Budget,
    FailureType,
    Heartbeat,
    JobEvent,
    JobState,
    MetricSnapshot,
    TrialConfig,
    TrialResult,
)
from autopretrain.monitor.anomaly_detector import AnomalyDetector, DetectorConfig
from autopretrain.orchestrator.state_machine import JobStateMachine, TrialState
from autopretrain.resilience.diagnoser import DiagnosisResult, MultiLayerDiagnoser
from autopretrain.resilience.recovery import AdaptiveRecoveryStrategy, RecoveryConfig
from autopretrain.resilience.self_diagnoser import SelfDiagnoser
from autopretrain.store.event_log import EventLog

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Polling
    poll_interval: float = 30.0
    heartbeat_check_interval: float = 60.0

    # Concurrency
    max_concurrent_trials: int = 3

    # Timeouts (dynamic — tightens after first successful step)
    initial_startup_timeout: int = 600  # 10 min for pod startup
    heartbeat_stale_threshold: int = 120  # 2 min
    heartbeat_dead_threshold: int = 360  # 6 min

    # Budget
    budget: Budget = field(default_factory=lambda: Budget(max_gpu_hours=100.0))

    # Paths
    heartbeat_dir: str = "/fsx/dev/jiaqi/experiments/autopretrain/heartbeat"
    metrics_dir: str = "/fsx/dev/jiaqi/experiments/autopretrain/live_metrics"
    event_log_path: str = "/fsx/dev/jiaqi/experiments/autopretrain/agent_logs/events.jsonl"
    knowledge_path: str = "/fsx/dev/jiaqi/experiments/autopretrain/agent_logs/learned_fixes.json"

    # Recovery
    recovery_config: RecoveryConfig = field(default_factory=RecoveryConfig)
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)

    # Auto-action threshold
    auto_action_confidence: float = 0.7

    # Infrastructure
    k8s_namespace: str = "default"
    k8s_context: str = "arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm"
    job_prefix: str = "luhanqin"


class Orchestrator:
    """Autonomous training orchestrator.

    Main loop:
        while not all_done and has_budget:
            1. Submit pending trials (up to concurrency limit)
            2. Read heartbeats for all running trials
            3. Check for anomalies in metrics
            4. Poll K8s status for state transitions
            5. Handle failures (diagnose → recover → resubmit)
            6. Update budget tracking
            7. Sleep poll_interval
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        compute_backend: Any,  # Will be ComputeBackend protocol
        training_engine: Any,  # Will be TrainingEngine protocol
    ) -> None:
        self.config = config
        self.compute = compute_backend
        self.engine = training_engine

        # Core components
        self.state_machine = JobStateMachine(on_event=self._on_event)
        self.diagnoser = MultiLayerDiagnoser()
        self.recovery = AdaptiveRecoveryStrategy(config.recovery_config)
        self.anomaly_detector = AnomalyDetector(config.detector_config)
        self.self_diagnoser = SelfDiagnoser(
            knowledge_path=Path(config.knowledge_path) if config.knowledge_path else None
        )
        self.event_log = EventLog(Path(config.event_log_path))

        # Runtime state
        self._metrics_history: dict[str, list[MetricSnapshot]] = {}
        self._running = False
        self._start_time: float = 0.0

    async def run(self, trials: list[TrialConfig]) -> list[TrialResult]:
        """Main entry point. Run all trials to completion.

        Args:
            trials: List of trial configurations to execute.

        Returns:
            List of TrialResult for completed trials.
        """
        self._running = True
        self._start_time = time.time()

        logger.info("Orchestrator starting with %d trials", len(trials))

        # Register all trials with state machine
        for trial in trials:
            self.state_machine.register_trial(trial)
            self._metrics_history[trial.trial_id] = []

        try:
            while self._running:
                # Check termination conditions
                if self.state_machine.all_done():
                    logger.info("All trials complete!")
                    break

                if not self.config.budget.has_budget:
                    logger.warning("Budget exhausted — stopping.")
                    await self._stop_all_running("Budget exhausted")
                    break

                # Main loop body
                await self._loop_iteration()

                # Sleep before next iteration
                await asyncio.sleep(self.config.poll_interval)

        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled.")
            await self._stop_all_running("Orchestrator cancelled")
        except Exception as e:
            logger.exception("Orchestrator error: %s", e)
            await self._stop_all_running(f"Orchestrator error: {e}")

        self._running = False
        return self._collect_results()

    def stop(self) -> None:
        """Signal graceful shutdown."""
        self._running = False

    async def _loop_iteration(self) -> None:
        """Single iteration of the main loop."""

        # Step 1: Submit pending trials (up to concurrency limit)
        await self._submit_pending_trials()

        # Step 2: Read heartbeats and check for silent hangs
        await self._check_heartbeats()

        # Step 3: Poll K8s status for state transitions
        await self._poll_job_statuses()

        # Step 4: Handle any trials in DIAGNOSING state
        await self._handle_diagnosing_trials()

        # Step 5: Handle any trials in RECOVERING state
        await self._handle_recovering_trials()

        # Step 6: Run anomaly detection on running trials
        await self._run_anomaly_detection()

    async def _submit_pending_trials(self) -> None:
        """Submit trials that are CREATED (not yet submitted)."""
        running_count = len(self.state_machine.get_running_trials())
        pending_states = [
            t for t in self.state_machine.get_all_trials()
            if t.state == JobState.CREATED
        ]

        for trial_state in pending_states:
            if running_count >= self.config.max_concurrent_trials:
                break

            await self._submit_trial(trial_state)
            running_count += 1

    async def _submit_trial(self, trial_state: TrialState) -> None:
        """Generate script + manifest and submit to K8s."""
        config = trial_state.config
        job_name = f"{self.config.job_prefix}-{config.trial_id}"

        try:
            # Delete any existing job with same name
            await self.compute.delete_job(job_name)
            await asyncio.sleep(2)

            # Generate training script and manifest
            script = self.engine.generate_script(config)
            manifest = self.engine.generate_manifest(config, script)

            # Submit
            success = await self.compute.submit_job(config, manifest)

            if success:
                trial_state.job_name = job_name
                self.state_machine.transition(
                    config.trial_id,
                    JobState.PENDING,
                    reason=f"Submitted as {job_name} (retry {config.current_retry})",
                )
            else:
                logger.error("Failed to submit %s", job_name)
                self.state_machine.transition(
                    config.trial_id,
                    JobState.FAILED,
                    reason="K8s submission failed",
                )
        except Exception as e:
            logger.exception("Error submitting trial %s: %s", config.trial_id, e)

    async def _check_heartbeats(self) -> None:
        """Check heartbeat files for all running trials."""
        for trial_state in self.state_machine.get_running_trials():
            heartbeat = await self._read_heartbeat(trial_state.config.trial_id)

            if heartbeat is None:
                # No heartbeat yet — check if startup timeout exceeded
                elapsed = time.time() - trial_state.submitted_at
                if elapsed > self.config.initial_startup_timeout:
                    self.state_machine.transition(
                        trial_state.trial_id,
                        JobState.DIAGNOSING,
                        reason=f"No heartbeat after {elapsed:.0f}s (startup timeout)",
                    )
                continue

            # Update state machine with heartbeat data
            self.state_machine.update_heartbeat(
                trial_state.trial_id,
                step=heartbeat.step,
                loss=heartbeat.loss,
            )

            # Check for dead heartbeat (silent hang)
            if heartbeat.is_dead:
                self.state_machine.transition(
                    trial_state.trial_id,
                    JobState.DIAGNOSING,
                    reason=f"Heartbeat dead ({heartbeat.age_seconds:.0f}s since step {heartbeat.step})",
                )

            # Collect metrics from heartbeat
            if heartbeat.loss is not None:
                snapshot = MetricSnapshot(
                    step=heartbeat.step,
                    timestamp=heartbeat.timestamp,
                    metrics={
                        "train/ce_loss": heartbeat.loss,
                        **({"optim/grad_norm": heartbeat.grad_norm} if heartbeat.grad_norm else {}),
                        **({"throughput/tokens_per_second": heartbeat.throughput_tps} if heartbeat.throughput_tps else {}),
                        **({"gpu_memory/pct": heartbeat.gpu_memory_pct} if heartbeat.gpu_memory_pct else {}),
                    },
                    job_id=trial_state.trial_id,
                )
                self._metrics_history[trial_state.trial_id].append(snapshot)

    async def _poll_job_statuses(self) -> None:
        """Poll K8s for job status changes."""
        for trial_state in self.state_machine.get_active_trials():
            if trial_state.state not in (JobState.PENDING, JobState.RUNNING):
                continue
            if not trial_state.job_name:
                continue

            try:
                status = await self.compute.get_job_status(trial_state.job_name)
            except Exception as e:
                logger.warning("Error polling %s: %s", trial_state.job_name, e)
                continue

            if status == "Running" and trial_state.state == JobState.PENDING:
                self.state_machine.transition(
                    trial_state.trial_id,
                    JobState.RUNNING,
                    reason="Pod entered Running state",
                )
            elif status == "Complete":
                self.state_machine.transition(
                    trial_state.trial_id,
                    JobState.COMPLETED,
                    reason="Job completed successfully",
                )
            elif status == "Failed":
                self.state_machine.transition(
                    trial_state.trial_id,
                    JobState.DIAGNOSING,
                    reason="K8s reports job Failed",
                )

    async def _handle_diagnosing_trials(self) -> None:
        """Run diagnosis on trials that have entered DIAGNOSING state."""
        for trial_state in self.state_machine.get_all_trials():
            if trial_state.state != JobState.DIAGNOSING:
                continue

            # Gather signals for diagnosis
            logs = ""
            events = ""
            heartbeat = None

            if trial_state.job_name:
                try:
                    logs = await self.compute.get_pod_logs(trial_state.job_name, tail=200)
                    events = await self.compute.get_pod_events(trial_state.job_name)
                except Exception as e:
                    logger.warning("Error getting logs for %s: %s", trial_state.job_name, e)

            heartbeat = await self._read_heartbeat(trial_state.trial_id)

            # Count concurrent failures for correlation
            concurrent = sum(
                1 for t in self.state_machine.get_all_trials()
                if t.state == JobState.DIAGNOSING and t.trial_id != trial_state.trial_id
            )

            # Run multi-layer diagnosis
            result = self.diagnoser.diagnose(
                logs=logs,
                events=events,
                heartbeat=heartbeat,
                history=trial_state.events,
                concurrent_failures=concurrent,
            )

            logger.info(
                "Diagnosis for %s: %s (confidence=%.2f)",
                trial_state.trial_id,
                result.failure_type.value,
                result.confidence,
            )

            # If UNKNOWN with low confidence, try self-diagnosis
            if result.failure_type == FailureType.UNKNOWN and result.confidence < 0.5:
                self_fix = await self.self_diagnoser.analyze_unknown_failure(
                    logs=logs,
                    events=events,
                    trial_config=trial_state.config,
                    history=trial_state.events,
                )
                if self_fix:
                    is_safe = await self.self_diagnoser.validate_fix(self_fix, trial_state.config)
                    if is_safe:
                        logger.info("Self-diagnoser proposed fix: %s", self_fix.description)
                        # Apply the self-diagnosed fix directly
                        self._apply_config_changes(trial_state.config, self_fix.config_changes)
                        self.state_machine.transition(
                            trial_state.trial_id,
                            JobState.RECOVERING,
                            reason=f"Self-diagnosed: {self_fix.description}",
                            details={"fix": self_fix.config_changes},
                        )
                        continue

            # Select recovery action from standard strategy
            node_name = None
            if trial_state.job_name:
                try:
                    node_name = await self.compute.get_node_name(trial_state.job_name)
                except Exception:
                    pass

            recovery = self.recovery.select_action(
                failure_type=result.failure_type,
                trial=trial_state.config,
                retry_count=trial_state.retry_count,
                node_name=node_name,
                budget=self.config.budget,
            )

            # Store diagnosis info
            details = {
                "failure_type": result.failure_type.value,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "recovery_action": recovery.action_type.value,
                "error_context": result.error_context[:500],
            }

            if recovery.action_type == ActionType.ALERT_HUMAN:
                self.state_machine.transition(
                    trial_state.trial_id,
                    JobState.ALERT_HUMAN,
                    reason=recovery.description,
                    details=details,
                )
            else:
                # Apply config changes and move to RECOVERING
                self._apply_config_changes(trial_state.config, recovery.config_changes)
                self.state_machine.transition(
                    trial_state.trial_id,
                    JobState.RECOVERING,
                    reason=recovery.description,
                    details=details,
                )

    async def _handle_recovering_trials(self) -> None:
        """Resubmit trials that are in RECOVERING state."""
        for trial_state in self.state_machine.get_all_trials():
            if trial_state.state != JobState.RECOVERING:
                continue

            # Increment retry count
            retry = self.state_machine.increment_retry(trial_state.trial_id)
            self.config.budget.consume_retry()

            # Check for checkpoint to resume from
            if trial_state.config.save_folder:
                ckpt = self.engine.find_latest_checkpoint(trial_state.config.save_folder)
                if ckpt:
                    trial_state.config.resume_from = trial_state.config.save_folder
                    logger.info("Will resume from checkpoint: %s", ckpt)

            # Move back to CREATED for resubmission
            # (RECOVERING → PENDING is the valid transition)
            self.state_machine.transition(
                trial_state.trial_id,
                JobState.PENDING,
                reason=f"Resubmitting (retry {retry})",
            )

            # Submit immediately
            await self._submit_trial(trial_state)

    async def _run_anomaly_detection(self) -> None:
        """Check for anomalies in running trials' metrics."""
        for trial_state in self.state_machine.get_running_trials():
            history = self._metrics_history.get(trial_state.trial_id, [])
            if not history:
                continue

            latest = history[-1]
            anomalies = self.anomaly_detector.check(latest, history[:-1])

            for anomaly in anomalies:
                logger.warning(
                    "Anomaly [%s step=%d]: %s (severity=%.2f)",
                    trial_state.trial_id,
                    anomaly.step,
                    anomaly.type.value,
                    anomaly.severity,
                )

                # High-severity anomalies trigger immediate action
                if anomaly.severity >= 0.9 and anomaly.type == AnomalyType.DIVERGENCE:
                    self.state_machine.transition(
                        trial_state.trial_id,
                        JobState.DIAGNOSING,
                        reason=f"Critical anomaly: {anomaly.message}",
                    )

    async def _stop_all_running(self, reason: str) -> None:
        """Stop all currently running trials."""
        for trial_state in self.state_machine.get_running_trials():
            if trial_state.job_name:
                try:
                    await self.compute.delete_job(trial_state.job_name)
                except Exception as e:
                    logger.warning("Error stopping %s: %s", trial_state.job_name, e)

    async def _read_heartbeat(self, trial_id: str) -> Heartbeat | None:
        """Read heartbeat file for a trial."""
        # This will be implemented by the HeartbeatReader
        # For now, return None (no heartbeat available)
        return None

    def _apply_config_changes(self, config: TrialConfig, changes: dict[str, Any]) -> None:
        """Apply recovery-recommended config changes to a trial."""
        for key, value in changes.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _collect_results(self) -> list[TrialResult]:
        """Collect results from all completed trials."""
        results = []
        for trial_state in self.state_machine.get_completed_trials():
            result = TrialResult(
                trial_id=trial_state.trial_id,
                config=trial_state.config,
                final_loss=trial_state.last_loss,
                final_step=trial_state.last_step,
                metrics_history=self._metrics_history.get(trial_state.trial_id, []),
                status="completed",
            )
            results.append(result)
        return results

    def _on_event(self, event: JobEvent) -> None:
        """Callback for all state transitions — logs to audit trail."""
        self.event_log.append(event)
