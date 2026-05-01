"""Promotion policy — decides whether a candidate becomes the new incumbent."""

from __future__ import annotations

from typing import Any

from ...types import TrainingSearchNode


class PromotionPolicy:
    """Promote a candidate if it beats the current incumbent's primary metric.

    Tie-breakers (spec §11):
      1. lower eval instability
      2. lower cost (seconds)
      3. lower output length (avg_output_tokens)
      4. fewer hard error buckets
    """

    def __init__(self) -> None:
        self.incumbent_id: str | None = None

    def update_incumbent(
        self,
        graph: Any,
        candidate: TrainingSearchNode,
    ) -> tuple[str | None, bool]:
        if candidate.is_valid is not True:
            return self.incumbent_id, False
        if candidate.metric is None:
            return self.incumbent_id, False

        if self.incumbent_id is None:
            self.incumbent_id = candidate.node_id
            return self.incumbent_id, True

        current = graph.nodes.get(self.incumbent_id)
        if current is None:
            self.incumbent_id = candidate.node_id
            return self.incumbent_id, True

        cand_metric = candidate.metric or 0.0
        curr_metric = current.metric or 0.0
        if cand_metric > curr_metric:
            self.incumbent_id = candidate.node_id
            return self.incumbent_id, True
        if cand_metric < curr_metric:
            return self.incumbent_id, False

        # Tie-break
        if _tie_break(candidate, current) > 0:
            self.incumbent_id = candidate.node_id
            return self.incumbent_id, True
        return self.incumbent_id, False


def _tie_break(candidate: TrainingSearchNode, current: TrainingSearchNode) -> int:
    """Return >0 if ``candidate`` beats ``current`` on tie-break rules."""

    def metric(node: TrainingSearchNode, key: str, default: float) -> float:
        buckets = node.error_buckets or {}
        cost = node.cost or {}
        secondary = {
            "instability": node.cost.get("instability", default) if node.cost else default,
            "seconds": cost.get("seconds", default),
            "avg_output_tokens": cost.get("avg_output_tokens", default),
            "hard_bucket_total": sum(
                v for k, v in buckets.items() if k in {"format_error", "wrong_rule", "overlong_reasoning"}
            ),
        }
        return float(secondary.get(key, default))

    for key in ("instability", "seconds", "avg_output_tokens", "hard_bucket_total"):
        cand = metric(candidate, key, 0.0)
        curr = metric(current, key, 0.0)
        if cand < curr:
            return 1
        if cand > curr:
            return -1
    return 0
