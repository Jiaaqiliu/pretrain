"""Decision engine for autonomous training actions.

Implements the core decision-making logic that connects monitoring signals
to concrete actions. Supports three autonomy levels:
- Full: act immediately on all decisions
- Semi: act on low-risk, ask for confirmation on high-risk
- Advisory: only suggest, never act
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from autopilot.agent.analyzer import ExperimentInsight
from autopilot.monitoring.anomaly import Severity, TrainingAnomaly
from autopilot.optimization.early_stopping import StopReason, StoppingDecision
from autopilot.utils.logging import get_logger
from autopilot.utils.persistence import DecisionRecord, StateStore

log = get_logger("agent.decision")


class ActionType(enum.Enum):
    EARLY_STOP = "early_stop"
    SKIP_STEP = "skip_step"
    ROLLBACK_CHECKPOINT = "rollback_checkpoint"
    REDUCE_LR = "reduce_lr"
    ADJUST_MIXTURE = "adjust_mixture"
    LAUNCH_EXPERIMENT = "launch_experiment"
    RESUME_EXPERIMENT = "resume_experiment"
    ADVANCE_PHASE = "advance_phase"
    NOTIFY_USER = "notify_user"
    NO_ACTION = "no_action"


class RiskLevel(enum.IntEnum):
    LOW = 1  # can execute immediately
    MEDIUM = 2  # execute in semi-auto, confirm in advisory
    HIGH = 3  # always confirm unless full-auto


@dataclass
class Action:
    """A concrete action to take on a training experiment."""

    action_type: ActionType
    experiment_id: str
    risk_level: RiskLevel
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


@dataclass
class DecisionContext:
    """All information available for making a decision."""

    experiment_id: str
    anomalies: List[TrainingAnomaly] = field(default_factory=list)
    stopping_decision: Optional[StoppingDecision] = None
    insights: List[ExperimentInsight] = field(default_factory=list)
    current_step: int = 0
    current_loss: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    """Makes autonomous decisions about training experiments.

    Decision flow:
    1. Receive context (anomalies, stopping signals, insights)
    2. Evaluate applicable rules
    3. Determine action + risk level
    4. Apply autonomy level filter
    5. Execute or queue for confirmation
    """

    def __init__(
        self,
        store: StateStore,
        autonomy_level: str = "semi",
        on_action: Optional[Callable[[Action], bool]] = None,
        on_confirmation_needed: Optional[Callable[[Action], bool]] = None,
    ):
        self._store = store
        self._autonomy = autonomy_level  # "full", "semi", "advisory"
        self._on_action = on_action
        self._on_confirmation_needed = on_confirmation_needed
        self._pending_actions: List[Action] = []
        self._action_history: List[Action] = []

    def decide(self, context: DecisionContext) -> List[Action]:
        """Evaluate context and produce actions."""
        actions: List[Action] = []

        # Handle critical anomalies (highest priority)
        for anomaly in context.anomalies:
            action = self._handle_anomaly(context, anomaly)
            if action:
                actions.append(action)

        # Handle stopping decisions
        if context.stopping_decision and context.stopping_decision.should_stop:
            action = self._handle_stopping(context, context.stopping_decision)
            if action:
                actions.append(action)

        # Handle insights
        for insight in context.insights:
            if insight.actionable and insight.suggested_action:
                action = self._handle_insight(context, insight)
                if action:
                    actions.append(action)

        # Deduplicate: prefer higher priority actions
        actions = self._deduplicate(actions)

        # Apply autonomy filter
        final_actions = []
        for action in actions:
            if self._should_execute(action):
                final_actions.append(action)
                self._record_decision(action)
            elif self._should_request_confirmation(action):
                action.requires_confirmation = True
                self._pending_actions.append(action)
                final_actions.append(action)

        return final_actions

    def confirm_action(self, action: Action) -> None:
        """User confirmed a pending action."""
        self._pending_actions = [a for a in self._pending_actions if a != action]
        self._record_decision(action)
        if self._on_action:
            self._on_action(action)

    def reject_action(self, action: Action) -> None:
        """User rejected a pending action."""
        self._pending_actions = [a for a in self._pending_actions if a != action]
        log.info(f"Action rejected: {action.action_type.value} for {action.experiment_id}")

    @property
    def pending_actions(self) -> List[Action]:
        return list(self._pending_actions)

    def _handle_anomaly(self, context: DecisionContext, anomaly: TrainingAnomaly) -> Optional[Action]:
        """Convert an anomaly into an action."""
        if anomaly.severity == Severity.CRITICAL:
            return Action(
                action_type=ActionType.ROLLBACK_CHECKPOINT,
                experiment_id=context.experiment_id,
                risk_level=RiskLevel.MEDIUM,
                reasoning=(
                    f"Critical anomaly: {anomaly.message}. "
                    f"Rolling back to last checkpoint."
                ),
                parameters={"rollback_steps": 100, "skip_batches": 200},
            )
        elif anomaly.severity == Severity.HIGH:
            if anomaly.anomaly_type.value == "gradient_explosion":
                return Action(
                    action_type=ActionType.REDUCE_LR,
                    experiment_id=context.experiment_id,
                    risk_level=RiskLevel.LOW,
                    reasoning=f"High gradient norm: {anomaly.message}. Reducing LR.",
                    parameters={"lr_factor": 0.5},
                )
            return Action(
                action_type=ActionType.SKIP_STEP,
                experiment_id=context.experiment_id,
                risk_level=RiskLevel.LOW,
                reasoning=f"Anomaly detected: {anomaly.message}. Skipping current step.",
            )
        elif anomaly.severity == Severity.MEDIUM:
            return Action(
                action_type=ActionType.NOTIFY_USER,
                experiment_id=context.experiment_id,
                risk_level=RiskLevel.LOW,
                reasoning=f"Medium anomaly: {anomaly.message}",
                parameters={"notification_type": "warning"},
            )
        return None

    def _handle_stopping(
        self, context: DecisionContext, decision: StoppingDecision
    ) -> Optional[Action]:
        """Convert a stopping decision into an action."""
        risk = RiskLevel.HIGH if decision.confidence < 0.8 else RiskLevel.MEDIUM

        if decision.reason in (StopReason.DIVERGING, StopReason.ASHA_PRUNED):
            risk = RiskLevel.MEDIUM

        return Action(
            action_type=ActionType.EARLY_STOP,
            experiment_id=context.experiment_id,
            risk_level=risk,
            reasoning=f"Early stopping recommended: {decision.message}",
            parameters={
                "reason": decision.reason.value if decision.reason else "unknown",
                "confidence": decision.confidence,
            },
        )

    def _handle_insight(
        self, context: DecisionContext, insight: ExperimentInsight
    ) -> Optional[Action]:
        """Convert an actionable insight into an action."""
        action_map = {
            "early_stop": ActionType.EARLY_STOP,
            "reduce_lr": ActionType.REDUCE_LR,
            "rollback_and_reduce_lr": ActionType.ROLLBACK_CHECKPOINT,
            "skip_step_and_monitor": ActionType.SKIP_STEP,
            "increase_lr_or_early_stop": ActionType.NOTIFY_USER,
            "reduce_grad_clip": ActionType.NOTIFY_USER,
        }

        action_type = action_map.get(insight.suggested_action, ActionType.NOTIFY_USER)

        return Action(
            action_type=action_type,
            experiment_id=context.experiment_id,
            risk_level=RiskLevel.MEDIUM,
            reasoning=f"Insight: {insight.description}",
            parameters={"source_insight": insight.title},
        )

    def _should_execute(self, action: Action) -> bool:
        """Determine if an action should be executed immediately."""
        if self._autonomy == "full":
            return True
        if self._autonomy == "semi":
            return action.risk_level == RiskLevel.LOW
        return False  # advisory mode: never auto-execute

    def _should_request_confirmation(self, action: Action) -> bool:
        """Determine if we should ask the user for confirmation."""
        if self._autonomy == "full":
            return False
        if self._autonomy == "semi":
            return action.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
        return True  # advisory: always show

    def _deduplicate(self, actions: List[Action]) -> List[Action]:
        """Remove duplicate actions, keeping highest priority."""
        seen = {}
        for action in actions:
            key = (action.experiment_id, action.action_type)
            if key not in seen or action.risk_level > seen[key].risk_level:
                seen[key] = action
        return list(seen.values())

    def _record_decision(self, action: Action) -> None:
        """Record a decision for audit trail."""
        record = DecisionRecord(
            decision_id=f"dec_{uuid.uuid4().hex[:8]}",
            experiment_id=action.experiment_id,
            decision_type=action.action_type.value,
            reasoning=action.reasoning,
            action=action.parameters,
        )
        self._store.save_decision(record)
        self._action_history.append(action)
