"""Reward policy for MCGS (search signal, NOT training loss)."""

from __future__ import annotations

from typing import Any

from ...types import (
    TrainingSearchNode,
    TrainingTrialResult,
    ValidityReport,
)


class DefaultRewardPolicy:
    """Implements the formula in mcgs_training_spec §8.

    reward =
        metric_delta
      - 0.10 * normalized_cost
      - 0.30 * format_error_rate
      - 0.10 * overlong_rate
      - 0.20 * instability_penalty

    If the trial is invalid, reward is -1.0.
    """

    def __init__(
        self,
        *,
        cost_weight: float = 0.10,
        format_weight: float = 0.30,
        overlong_weight: float = 0.10,
        instability_weight: float = 0.20,
        invalid_reward: float = -1.0,
    ):
        self.cost_weight = cost_weight
        self.format_weight = format_weight
        self.overlong_weight = overlong_weight
        self.instability_weight = instability_weight
        self.invalid_reward = invalid_reward

    def compute(
        self,
        trial_result: TrainingTrialResult,
        validity: ValidityReport,
        parent: TrainingSearchNode | None,
        graph: Any,  # noqa: ARG002 — graph kept for future policies
    ) -> float:
        if not validity.is_valid:
            return self.invalid_reward

        metrics = trial_result.eval_metrics
        if metrics is None:
            return self.invalid_reward

        parent_metric = parent.metric if (parent is not None and parent.metric is not None) else 0.0
        metric_delta = metrics.primary_metric_value - parent_metric

        cost_seconds = float(trial_result.cost.get("seconds", 0.0) or 0.0)
        # Normalize cost to [0, 1] assuming a 1-hour reference budget.
        normalized_cost = min(cost_seconds / 3600.0, 1.0)

        buckets = trial_result.error_buckets.counts if trial_result.error_buckets else {}
        total = sum(buckets.values()) or 1
        format_rate = buckets.get("format_error", 0) / total
        overlong_rate = buckets.get("overlong_reasoning", 0) / total

        instability = float(trial_result.train_metrics.get("instability", 0.0) or 0.0)
        instability = max(0.0, min(1.0, instability))

        return (
            metric_delta
            - self.cost_weight * normalized_cost
            - self.format_weight * format_rate
            - self.overlong_weight * overlong_rate
            - self.instability_weight * instability
        )
