"""Bits hint provider.

The bits puzzle is under-determined: many family-F rules fit the examples,
each producing a different question-bit. This provider runs the
*constrained* search: find any combination of per-bit rules that fits the
examples AND produces the Kaggle stored answer on the question. If no such
combination exists, ``compute_hint`` returns ``None`` (drop the row — the
Kaggle generator used a rule outside family F; we measured 14/1602 such
rows).

Reuses ``is_bits_label_consistent`` from the verifier — that function
already runs the constrained search and returns the per-bit witnesses; we
just package them in the ``Hint`` shape.
"""

from __future__ import annotations

from agent_evolve.model.data.verifiers.bits import (
    _parse_bits_prompt,
    is_bits_label_consistent,
)

from ._protocol import Hint


def compute_hint(prompt: str, kaggle_answer: str) -> Hint | None:
    pairs, q = _parse_bits_prompt(prompt)
    if not pairs or not q:
        return None
    consistent, witnesses = is_bits_label_consistent(pairs, q, kaggle_answer)
    if not consistent:
        return None  # Kaggle uses a rule outside family F — drop
    # Compact human-readable summary
    if len(set(witnesses)) == 1:
        rule_summary = f"global byte transform: {witnesses[0]}"
    else:
        rule_summary = "per-bit rules (see per_component)"
    return Hint(
        rule_summary=rule_summary,
        per_component=list(witnesses),
        applies_cleanly=True,
        extras={
            "examples": pairs,
            "question": q,
            "kaggle_answer": kaggle_answer,
        },
    )


__all__ = ["compute_hint"]
