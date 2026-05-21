"""Early stopping strategies for LLM training experiments.

Implements multiple early stopping criteria:
- ASHA (Asynchronous Successive Halving): prune worst-performing experiments at each rung
- Predictive stopping: stop if predicted final loss exceeds threshold
- Plateau detection: stop if no improvement over N steps
- Resource-aware stopping: balance quality vs. compute cost
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


from autopilot.monitoring.metrics import MetricsWindow
from autopilot.monitoring.prediction import LossPredictor
from autopilot.utils.logging import get_logger

log = get_logger("optimization.early_stopping")


class StopReason(enum.Enum):
    ASHA_PRUNED = "asha_pruned"
    PREDICTED_UNDERPERFORM = "predicted_underperform"
    PLATEAU = "plateau"
    DIVERGING = "diverging"
    RESOURCE_LIMIT = "resource_limit"
    TARGET_REACHED = "target_reached"


@dataclass
class StoppingDecision:
    """The result of an early stopping check."""

    should_stop: bool
    reason: Optional[StopReason] = None
    confidence: float = 0.0
    message: str = ""
    details: Dict = field(default_factory=dict)


@dataclass
class ASHAConfig:
    """Configuration for ASHA successive halving."""

    min_resource: int = 500  # minimum steps before first evaluation
    max_resource: int = 100000  # maximum training steps
    reduction_factor: int = 3  # keep top 1/3 at each rung
    num_rungs: int = 4


class EarlyStoppingStrategy:
    """Combined early stopping strategy.

    Evaluates multiple stopping criteria and returns the strongest signal.
    """

    def __init__(
        self,
        total_steps: int,
        target_loss: Optional[float] = None,
        patience: int = 2000,
        min_improvement: float = 0.001,
        asha_config: Optional[ASHAConfig] = None,
    ):
        self._total_steps = total_steps
        self._target_loss = target_loss
        self._patience = patience
        self._min_improvement = min_improvement
        self._asha = asha_config or ASHAConfig(max_resource=total_steps)
        self._predictor = LossPredictor(total_steps=total_steps, target_loss=target_loss)
        self._best_loss: Dict[str, float] = {}
        self._best_step: Dict[str, int] = {}

    def should_stop(self, experiment_id: str, window: MetricsWindow) -> StoppingDecision:
        """Check if an experiment should be stopped."""
        if window.length < 100:
            return StoppingDecision(should_stop=False)

        latest = window.latest
        if latest is None or "loss" not in latest.metrics:
            return StoppingDecision(should_stop=False)

        current_loss = latest.metrics["loss"]
        current_step = latest.step

        # Check target reached
        if self._target_loss and current_loss <= self._target_loss:
            return StoppingDecision(
                should_stop=True,
                reason=StopReason.TARGET_REACHED,
                confidence=1.0,
                message=f"Target loss {self._target_loss:.4f} reached at step {current_step}",
            )

        # Check divergence
        trend = window.trend("loss", last_n=100)
        if trend is not None and trend > 0.01:
            return StoppingDecision(
                should_stop=True,
                reason=StopReason.DIVERGING,
                confidence=0.8,
                message=f"Loss is diverging (trend={trend:.4f}/step)",
                details={"trend": trend},
            )

        # Check plateau
        plateau_decision = self._check_plateau(experiment_id, current_loss, current_step)
        if plateau_decision.should_stop:
            return plateau_decision

        # Predictive stopping
        predictive_decision = self._check_predicted_underperform(window)
        if predictive_decision.should_stop:
            return predictive_decision

        return StoppingDecision(should_stop=False)

    def asha_evaluate(
        self, experiments: Dict[str, MetricsWindow], rung_step: int
    ) -> List[Tuple[str, StoppingDecision]]:
        """ASHA-style evaluation at a rung: prune worst-performing experiments.

        Returns list of (experiment_id, StoppingDecision) for experiments that should stop.
        """
        eligible = []
        for eid, window in experiments.items():
            latest = window.latest
            if latest and latest.step >= rung_step and "loss" in latest.metrics:
                eligible.append((eid, latest.metrics["loss"]))

        if len(eligible) <= 1:
            return []

        # Sort by loss (ascending = better)
        eligible.sort(key=lambda x: x[1])

        # Keep top 1/reduction_factor
        n_keep = max(1, len(eligible) // self._asha.reduction_factor)
        to_prune = eligible[n_keep:]

        decisions = []
        for eid, loss in to_prune:
            decisions.append(
                (
                    eid,
                    StoppingDecision(
                        should_stop=True,
                        reason=StopReason.ASHA_PRUNED,
                        confidence=0.7,
                        message=(
                            f"ASHA pruned at rung {rung_step}: "
                            f"loss={loss:.4f} (rank {eligible.index((eid, loss))+1}/{len(eligible)})"
                        ),
                        details={
                            "rung": rung_step,
                            "rank": eligible.index((eid, loss)) + 1,
                            "total": len(eligible),
                            "cutoff_loss": eligible[n_keep - 1][1],
                        },
                    ),
                )
            )

        return decisions

    def get_asha_rungs(self) -> List[int]:
        """Compute ASHA rung steps (evaluation checkpoints)."""
        rungs = []
        resource = self._asha.min_resource
        for _ in range(self._asha.num_rungs):
            rungs.append(resource)
            resource *= self._asha.reduction_factor
            if resource > self._asha.max_resource:
                break
        return rungs

    def _check_plateau(
        self, experiment_id: str, current_loss: float, current_step: int
    ) -> StoppingDecision:
        best = self._best_loss.get(experiment_id, float("inf"))
        if current_loss < best - self._min_improvement:
            self._best_loss[experiment_id] = current_loss
            self._best_step[experiment_id] = current_step
            return StoppingDecision(should_stop=False)

        steps_since_best = current_step - self._best_step.get(experiment_id, 0)
        if steps_since_best > self._patience:
            return StoppingDecision(
                should_stop=True,
                reason=StopReason.PLATEAU,
                confidence=0.6,
                message=(
                    f"No improvement for {steps_since_best} steps "
                    f"(best={best:.4f} at step {self._best_step[experiment_id]})"
                ),
                details={
                    "best_loss": best,
                    "best_step": self._best_step[experiment_id],
                    "steps_without_improvement": steps_since_best,
                },
            )

        return StoppingDecision(should_stop=False)

    def _check_predicted_underperform(self, window: MetricsWindow) -> StoppingDecision:
        if self._target_loss is None:
            return StoppingDecision(should_stop=False)

        outlook = self._predictor.predict(window)
        if outlook is None:
            return StoppingDecision(should_stop=False)

        if not outlook.on_track and outlook.confidence > 0.7:
            margin = (outlook.predicted_final_loss - self._target_loss) / self._target_loss
            if margin > 0.2:  # predicted to exceed target by >20%
                return StoppingDecision(
                    should_stop=True,
                    reason=StopReason.PREDICTED_UNDERPERFORM,
                    confidence=outlook.confidence,
                    message=(
                        f"Predicted final loss ({outlook.predicted_final_loss:.4f}) "
                        f"exceeds target ({self._target_loss:.4f}) by {margin*100:.1f}%"
                    ),
                    details={
                        "predicted_final": outlook.predicted_final_loss,
                        "target": self._target_loss,
                        "confidence": outlook.confidence,
                    },
                )

        return StoppingDecision(should_stop=False)
