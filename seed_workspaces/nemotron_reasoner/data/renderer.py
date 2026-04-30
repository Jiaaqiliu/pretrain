"""Dataset renderer for the nemotron_reasoner seed workspace.

The real implementation would tokenize prompts + targets using the model's
tokenizer. In smoke mode the TinkerLite backend falls back to ``render_datums``
in ``agent_evolve.training.runners.helpers.dataset``, which treats integer IDs
directly.
"""

from __future__ import annotations

from typing import Iterable


def render(rows: Iterable[dict]) -> list[dict]:
    """Return the rows unchanged — smoke renderer is a pass-through."""
    return list(rows)
