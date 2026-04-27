"""Placeholder reward function for the RL stage (disabled in v1 seed)."""

from __future__ import annotations


def compute_reward(prompt: str, completion: str, reference: str | None = None) -> float:
    """Return 1.0 if completion exactly matches the reference, else 0.0."""
    if reference is None:
        return 0.0
    return 1.0 if completion.strip() == reference.strip() else 0.0
