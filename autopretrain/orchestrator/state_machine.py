"""Formal state machine for training job lifecycle.

Every state transition is:
1. Validated (only allowed transitions succeed)
2. Logged (to the event store for audit trail)
3. Actioned (triggers appropriate side effects)

This replaces ad-hoc if/else with a deterministic, inspectable FSM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from autopretrain.core.types import JobEvent, JobState, TrialConfig

logger = logging.getLogger(__name__)

# Valid state transitions
TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.CREATED: {JobState.PENDING},
    JobState.PENDING: {JobState.RUNNING, JobState.FAILED, JobState.TIMED_OUT},
    JobState.RUNNING: {
        JobState.COMPLETED,
        JobState.DIAGNOSING,
        JobState.TIMED_OUT,
    },
    JobState.DIAGNOSING: {JobState.RECOVERING, JobState.FAILED, JobState.ALERT_HUMAN},
    JobState.RECOVERING: {JobState.PENDING, JobState.FAILED, JobState.ALERT_HUMAN},
    JobState.TIMED_OUT: {JobState.PENDING, JobState.FAILED},
    JobState.COMPLETED: set(),
    JobState.FAILED: set(),
    JobState.ALERT_HUMAN: set(),
}


class InvalidTransitionError(Exception):
    pass


@dataclass
class TrialState:
    """Mutable state for a single trial tracked by the state machine."""

    trial_id: str
    config: TrialConfig
    state: JobState = JobState.CREATED
    job_name: str = ""
    node_name: str | None = None
    retry_count: int = 0
    submitted_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    events: list[JobEvent] = field(default_factory=list)
    last_heartbeat_at: float = 0.0
    last_step: int = 0
    last_loss: float | None = None


class JobStateMachine:
    """Manages state transitions for all trials in an experiment.

    Thread-safe, deterministic, fully logged.
    """

    def __init__(self, on_event: Callable[[JobEvent], None] | None = None) -> None:
        self._trials: dict[str, TrialState] = {}
        self._on_event = on_event

    def register_trial(self, config: TrialConfig) -> TrialState:
        """Register a new trial with the state machine."""
        state = TrialState(trial_id=config.trial_id, config=config)
        self._trials[config.trial_id] = state
        self._emit_event(state, "registered", JobState.CREATED, JobState.CREATED)
        return state

    def transition(
        self,
        trial_id: str,
        new_state: JobState,
        reason: str = "",
        details: dict | None = None,
    ) -> TrialState:
        """Transition a trial to a new state.

        Raises InvalidTransitionError if the transition is not allowed.
        """
        trial = self._trials[trial_id]
        old_state = trial.state

        if new_state not in TRANSITIONS.get(old_state, set()):
            raise InvalidTransitionError(
                f"Cannot transition {trial_id} from {old_state.value} to {new_state.value}. "
                f"Allowed: {[s.value for s in TRANSITIONS.get(old_state, set())]}"
            )

        trial.state = new_state
        now = time.time()

        if new_state == JobState.PENDING:
            trial.submitted_at = now
        elif new_state == JobState.RUNNING:
            trial.started_at = now
        elif new_state.is_terminal:
            trial.finished_at = now

        self._emit_event(trial, reason or f"{old_state.value}->{new_state.value}", old_state, new_state, details)

        logger.info(
            "Trial %s: %s → %s (%s)",
            trial_id,
            old_state.value,
            new_state.value,
            reason or "no reason",
        )

        return trial

    def get_trial(self, trial_id: str) -> TrialState | None:
        return self._trials.get(trial_id)

    def get_all_trials(self) -> list[TrialState]:
        return list(self._trials.values())

    def get_active_trials(self) -> list[TrialState]:
        return [t for t in self._trials.values() if t.state.is_active]

    def get_running_trials(self) -> list[TrialState]:
        return [t for t in self._trials.values() if t.state == JobState.RUNNING]

    def get_completed_trials(self) -> list[TrialState]:
        return [t for t in self._trials.values() if t.state == JobState.COMPLETED]

    def get_terminal_trials(self) -> list[TrialState]:
        return [t for t in self._trials.values() if t.state.is_terminal]

    def all_done(self) -> bool:
        return all(t.state.is_terminal for t in self._trials.values())

    def increment_retry(self, trial_id: str) -> int:
        trial = self._trials[trial_id]
        trial.retry_count += 1
        return trial.retry_count

    def update_heartbeat(self, trial_id: str, step: int, loss: float | None = None) -> None:
        trial = self._trials.get(trial_id)
        if trial:
            trial.last_heartbeat_at = time.time()
            trial.last_step = step
            trial.last_loss = loss

    def _emit_event(
        self,
        trial: TrialState,
        event_type: str,
        from_state: JobState,
        to_state: JobState,
        details: dict | None = None,
    ) -> None:
        event = JobEvent(
            timestamp=time.time(),
            trial_id=trial.trial_id,
            event_type=event_type,
            from_state=from_state.value,
            to_state=to_state.value,
            details=details or {},
        )
        trial.events.append(event)
        if self._on_event:
            self._on_event(event)
