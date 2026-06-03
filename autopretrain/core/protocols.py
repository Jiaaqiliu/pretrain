"""Protocol interfaces defining contracts between layers.

Each layer communicates through these protocols, enabling:
- Pluggable backends (K8s, SLURM, local)
- Pluggable training engines (OLMo-core, NeMo, etc.)
- Testability (mock any layer in isolation)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from autopretrain.core.types import (
    Anomaly,
    Heartbeat,
    JobEvent,
    MetricSnapshot,
    RecoveryAction,
    TrialConfig,
    FailureType,
)


@runtime_checkable
class ComputeBackend(Protocol):
    """Manages compute resources and job lifecycle on a cluster."""

    async def submit_job(self, trial: TrialConfig, manifest: str) -> str:
        """Submit a training job. Returns the job name/ID."""
        ...

    async def delete_job(self, job_name: str) -> bool:
        """Delete/cancel a running job."""
        ...

    async def get_job_status(self, job_name: str) -> str:
        """Get job status: Running, Complete, Failed, Pending, NotFound."""
        ...

    async def get_pod_logs(self, job_name: str, tail: int = 200) -> str:
        """Get recent logs from the job's pod."""
        ...

    async def get_pod_events(self, job_name: str) -> str:
        """Get K8s events related to this job."""
        ...

    async def get_node_name(self, job_name: str) -> str | None:
        """Get the node a job is running on (for node exclusion)."""
        ...


@runtime_checkable
class TrainingEngine(Protocol):
    """Generates training scripts and reads results from a training framework."""

    def generate_script(self, trial: TrialConfig) -> str:
        """Generate a self-contained training script for this trial."""
        ...

    def generate_manifest(self, trial: TrialConfig, script_path: str) -> str:
        """Generate a K8s Job manifest for this trial."""
        ...

    def read_metrics(self, save_folder: str) -> list[MetricSnapshot]:
        """Read training metrics from a completed trial's save folder."""
        ...

    def find_latest_checkpoint(self, save_folder: str) -> str | None:
        """Find the latest valid checkpoint in a save folder."""
        ...


@runtime_checkable
class HeartbeatReader(Protocol):
    """Reads heartbeat signals from running trials."""

    async def read_heartbeat(self, trial_id: str) -> Heartbeat | None:
        """Read the latest heartbeat for a trial. None if no heartbeat exists."""
        ...

    async def list_active_heartbeats(self) -> list[Heartbeat]:
        """List all heartbeats that are not dead."""
        ...


@runtime_checkable
class FailureDiagnoser(Protocol):
    """Diagnoses root cause of job failures."""

    def diagnose(
        self,
        logs: str,
        events: str,
        heartbeat: Heartbeat | None,
        history: list[JobEvent],
    ) -> FailureType:
        """Classify failure type from available signals."""
        ...

    def get_error_context(self, logs: str, max_lines: int = 30) -> str:
        """Extract relevant error context from logs."""
        ...


@runtime_checkable
class RecoveryStrategy(Protocol):
    """Selects recovery actions based on failure diagnosis."""

    def select_action(
        self,
        failure_type: FailureType,
        trial: TrialConfig,
        retry_count: int,
        node_name: str | None = None,
    ) -> RecoveryAction:
        """Select the best recovery action for this failure."""
        ...


@runtime_checkable
class AnomalyDetector(Protocol):
    """Detects training anomalies from metric snapshots."""

    def check(
        self,
        snapshot: MetricSnapshot,
        history: list[MetricSnapshot],
    ) -> list[Anomaly]:
        """Check for anomalies given current snapshot and history."""
        ...


@runtime_checkable
class EventStore(Protocol):
    """Persistent storage for job events (audit trail)."""

    def append(self, event: JobEvent) -> None:
        """Append an event to the log."""
        ...

    def query(
        self,
        trial_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[JobEvent]:
        """Query stored events."""
        ...


@runtime_checkable
class Notifier(Protocol):
    """Sends notifications about important events."""

    async def notify(self, level: str, message: str, details: dict[str, Any] | None = None) -> None:
        """Send a notification. Level: info, warning, error, critical."""
        ...


@runtime_checkable
class SelfDiagnoser(Protocol):
    """Self-diagnosis module that can analyze and fix issues autonomously.

    This is the 'self-evolving' layer — when standard recovery fails,
    the SelfDiagnoser uses deeper analysis (and optionally LLM reasoning)
    to understand novel failures and propose fixes.
    """

    async def analyze_unknown_failure(
        self,
        logs: str,
        events: str,
        trial_config: TrialConfig,
        history: list[JobEvent],
    ) -> RecoveryAction | None:
        """Analyze an unknown failure and propose a fix.

        Returns None if the failure cannot be diagnosed autonomously.
        """
        ...

    async def validate_fix(
        self,
        proposed_action: RecoveryAction,
        trial_config: TrialConfig,
    ) -> bool:
        """Validate that a proposed fix is safe to apply.

        Returns True if the fix passes safety checks.
        """
        ...

    def learn_from_outcome(
        self,
        failure_type: FailureType,
        action_taken: RecoveryAction,
        success: bool,
    ) -> None:
        """Record outcome for future learning.

        Successful fixes for previously-unknown failures get added to
        the known pattern database for instant matching next time.
        """
        ...
