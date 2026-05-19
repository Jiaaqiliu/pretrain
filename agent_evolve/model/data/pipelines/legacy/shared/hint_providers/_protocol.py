"""Hint provider protocol — uniform interface for per-domain rule extraction.

Each domain implements ``compute_hint(prompt, kaggle_answer)`` returning a
``Hint`` (or ``None`` if no rule in the domain's family produces the stored
answer — caller drops the row).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Hint:
    """A rule (or set of per-component rules) consistent with the examples
    AND producing ``kaggle_answer`` on the question.

    ``rule_summary``: one-line description of the chosen rule (e.g.
    ``"AND(I0, I1)"``, ``"output = 0.65 × input"``).

    ``per_component``: optional list of finer-grained rules — for bits this
    is 8 per-bit rule names; for equations it's per-operator rules; for
    well-determined domains it can be empty.

    ``applies_cleanly``: True iff the hint was derived without any
    fall-back search. Verifier-as-oracle domains (numerals/units/gravity/
    cipher) always set this True. Family-membership domains (bits/
    equations) set it True if the *first-pass* solver agreed with the
    stored answer, False if the constrained re-search had to invent a
    different rule.

    ``extras``: per-domain payload the prompt template can interpolate
    (e.g. arithmetic intermediates, witness tables).
    """

    rule_summary: str
    per_component: list[str] = field(default_factory=list)
    applies_cleanly: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


class HintProvider(Protocol):
    """Each domain module exposes ``compute_hint`` matching this signature."""

    def compute_hint(
        self, prompt: str, kaggle_answer: str
    ) -> Hint | None: ...


__all__ = ["Hint", "HintProvider"]
