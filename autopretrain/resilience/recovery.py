"""Recovery strategy selection with adaptive behavior.

Maps failure types to recovery actions using an event-action table
(inspired by Volcano's task-level policies). Adapts strategies based
on retry history — e.g., first OOM halves batch, second enables grad ckpt.

Key design principles:
- Deterministic: same inputs → same output (no randomness in recovery)
- Escalating: each retry tries a more aggressive fix
- Safe: non-retryable errors never get retried
- Budget-aware: checks remaining budget before approving retry
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from autopretrain.core.types import (
    ActionType,
    Budget,
    FailureType,
    RecoveryAction,
    TrialConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""

    max_retries: int = 5
    base_backoff_seconds: int = 30
    max_backoff_seconds: int = 600
    oom_batch_reduction_factor: float = 0.5
    lr_reduction_factor: float = 0.5
    network_base_wait: int = 30
    preemption_wait: int = 60
    resource_wait: int = 300
    unknown_max_retries: int = 2


class AdaptiveRecoveryStrategy:
    """Selects recovery actions with escalating fixes per retry count."""

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()
        self._excluded_nodes: set[str] = set()
        self._node_failure_counts: dict[str, int] = {}

    def select_action(
        self,
        failure_type: FailureType,
        trial: TrialConfig,
        retry_count: int,
        node_name: str | None = None,
        budget: Budget | None = None,
    ) -> RecoveryAction:
        """Select the best recovery action given all context.

        Args:
            failure_type: Diagnosed failure type
            trial: Current trial config
            retry_count: How many times this trial has already been retried
            node_name: Which node it was running on (for exclusion)
            budget: Remaining budget (to check if retry is worthwhile)
        """
        # Budget check
        if budget and not budget.has_budget:
            return RecoveryAction(
                action_type=ActionType.ALERT_HUMAN,
                description="Budget exhausted — cannot retry",
                confidence=1.0,
            )

        # Max retries check
        if retry_count >= self.config.max_retries:
            return RecoveryAction(
                action_type=ActionType.ALERT_HUMAN,
                description=f"Max retries ({retry_count}/{self.config.max_retries}) exceeded",
                confidence=1.0,
            )

        # Non-retryable failures
        if not failure_type.is_retryable:
            return RecoveryAction(
                action_type=ActionType.ALERT_HUMAN,
                description=f"Non-retryable failure: {failure_type.value}",
                confidence=1.0,
            )

        # Track node failures for exclusion
        if node_name:
            self._node_failure_counts[node_name] = self._node_failure_counts.get(node_name, 0) + 1
            if self._node_failure_counts[node_name] >= 2:
                self._excluded_nodes.add(node_name)
                logger.warning("Node %s excluded after %d failures", node_name, self._node_failure_counts[node_name])

        # Dispatch to type-specific handler
        handlers = {
            FailureType.NETWORK_DISCONNECT: self._handle_network,
            FailureType.OOM: self._handle_oom,
            FailureType.NCCL_TIMEOUT: self._handle_nccl,
            FailureType.PREEMPTION: self._handle_preemption,
            FailureType.SILENT_HANG: self._handle_silent_hang,
            FailureType.LOSS_DIVERGENCE: self._handle_divergence,
            FailureType.NAN_DETECTED: self._handle_nan,
            FailureType.STRAGGLER: self._handle_straggler,
            FailureType.RESOURCE_UNAVAILABLE: self._handle_resources,
            FailureType.NODE_REPEATED_FAIL: self._handle_node_fail,
            FailureType.DEADLINE_EXCEEDED: self._handle_deadline,
            FailureType.UNKNOWN: self._handle_unknown,
        }

        handler = handlers.get(failure_type, self._handle_unknown)
        return handler(trial, retry_count, node_name)

    @property
    def excluded_nodes(self) -> set[str]:
        return self._excluded_nodes.copy()

    def _backoff(self, retry_count: int, base: int | None = None) -> int:
        """Exponential backoff with jitter ceiling."""
        b = base or self.config.base_backoff_seconds
        delay = min(b * (2 ** retry_count), self.config.max_backoff_seconds)
        return delay

    def _handle_network(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        return RecoveryAction(
            action_type=ActionType.RESUBMIT,
            description=f"Network disconnect (transient) — resubmit after {self._backoff(retry, self.config.network_base_wait)}s",
            wait_seconds=self._backoff(retry, self.config.network_base_wait),
            confidence=0.95,
        )

    def _handle_oom(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        current_microbatch = trial.rank_microbatch_size

        if retry == 0:
            new_microbatch = max(1, int(current_microbatch * self.config.oom_batch_reduction_factor))
            return RecoveryAction(
                action_type=ActionType.REDUCE_BATCH,
                description=f"OOM — reduce microbatch {current_microbatch} → {new_microbatch}",
                config_changes={"rank_microbatch_size": new_microbatch},
                confidence=0.9,
            )
        elif retry == 1:
            return RecoveryAction(
                action_type=ActionType.ENABLE_GRAD_CKPT,
                description="OOM persists — enable gradient checkpointing",
                config_changes={"enable_gradient_checkpointing": True},
                confidence=0.85,
            )
        elif retry == 2:
            new_microbatch = max(1, int(current_microbatch * self.config.oom_batch_reduction_factor ** 2))
            return RecoveryAction(
                action_type=ActionType.REDUCE_BATCH,
                description=f"OOM persists — aggressive batch reduction to {new_microbatch}",
                config_changes={
                    "rank_microbatch_size": new_microbatch,
                    "enable_gradient_checkpointing": True,
                },
                confidence=0.75,
            )
        else:
            return RecoveryAction(
                action_type=ActionType.ALERT_HUMAN,
                description="OOM persists after 3 escalating fixes — model may be too large for hardware",
                confidence=1.0,
            )

    def _handle_nccl(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        wait = self._backoff(retry, 60)
        return RecoveryAction(
            action_type=ActionType.RESUBMIT,
            description=f"NCCL timeout — resubmit after {wait}s (often transient)",
            wait_seconds=wait,
            confidence=0.9,
        )

    def _handle_preemption(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        return RecoveryAction(
            action_type=ActionType.RESUBMIT,
            description="Node preempted — resubmit with checkpoint resume",
            wait_seconds=self.config.preemption_wait,
            confidence=0.95,
        )

    def _handle_silent_hang(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        if node:
            self._excluded_nodes.add(node)
        return RecoveryAction(
            action_type=ActionType.FORCE_KILL,
            description="Silent hang detected — kill and resubmit",
            wait_seconds=10,
            confidence=0.85,
        )

    def _handle_divergence(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        new_lr = trial.lr * self.config.lr_reduction_factor
        return RecoveryAction(
            action_type=ActionType.ROLLBACK,
            description=f"Loss diverging — rollback checkpoint + reduce lr to {new_lr:.2e}",
            config_changes={"lr": new_lr, "rollback_steps": 1000},
            confidence=0.8,
        )

    def _handle_nan(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        new_lr = trial.lr * self.config.lr_reduction_factor
        return RecoveryAction(
            action_type=ActionType.ROLLBACK,
            description=f"NaN detected — rollback + reduce lr to {new_lr:.2e}",
            config_changes={"lr": new_lr},
            confidence=0.9,
        )

    def _handle_straggler(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        if retry == 0:
            return RecoveryAction(
                action_type=ActionType.WAIT,
                description="Straggler detected — waiting 5min for recovery",
                wait_seconds=300,
                confidence=0.6,
            )
        if node:
            self._excluded_nodes.add(node)
        return RecoveryAction(
            action_type=ActionType.RESUBMIT,
            description="Straggler persists — resubmit (node excluded)",
            config_changes={"exclude_node": node},
            confidence=0.7,
        )

    def _handle_resources(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        return RecoveryAction(
            action_type=ActionType.WAIT,
            description=f"Resources unavailable — waiting {self.config.resource_wait}s",
            wait_seconds=self.config.resource_wait,
            confidence=0.8,
        )

    def _handle_node_fail(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        if node:
            self._excluded_nodes.add(node)
        return RecoveryAction(
            action_type=ActionType.RESUBMIT,
            description=f"Repeated node failure — node excluded, resubmit",
            confidence=0.9,
        )

    def _handle_deadline(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        return RecoveryAction(
            action_type=ActionType.ALERT_HUMAN,
            description="Deadline exceeded — trial took too long",
            confidence=0.95,
        )

    def _handle_unknown(self, trial: TrialConfig, retry: int, node: str | None) -> RecoveryAction:
        if retry < self.config.unknown_max_retries:
            wait = self._backoff(retry, 60)
            return RecoveryAction(
                action_type=ActionType.RESUBMIT,
                description=f"Unknown failure — retry {retry + 1}/{self.config.unknown_max_retries} after {wait}s",
                wait_seconds=wait,
                confidence=0.5,
            )
        return RecoveryAction(
            action_type=ActionType.ALERT_HUMAN,
            description=f"Unknown failure persists after {retry} retries — needs human investigation",
            confidence=1.0,
        )
