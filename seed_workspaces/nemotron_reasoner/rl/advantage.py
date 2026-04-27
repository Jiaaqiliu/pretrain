"""Placeholder advantage function (disabled in v1 seed)."""

from __future__ import annotations


def compute_advantages(rewards: list[float], baseline: float = 0.0) -> list[float]:
    return [r - baseline for r in rewards]
