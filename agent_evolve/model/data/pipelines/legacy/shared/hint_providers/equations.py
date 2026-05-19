"""Equations hint provider.

Equations is structurally like bits: the puzzle is *under-determined*,
many rules in family F (S1–S6 + arithmetic z3) fit the examples but only
some land at the Kaggle stored answer on the question. This provider
runs the same constrained search the verifier already implements, then
packages the witness so the teacher can write a coherent CoT around it.

Reuses ``verify`` from the verifier — that function already runs the
arithmetic-z3 + symbolic S1–S6 cascade and returns a witness dict; we
just shape it into the ``Hint`` contract.

Drop policy: when the search returns ``status != "ok"``, no rule in
family F lands at the stored answer. Caller drops the row (the bits
pipeline measured 14/1602 such rows; equations is expected to drop more
because the puzzle distribution is harder).
"""

from __future__ import annotations

from agent_evolve.model.data.verifiers.equations import (
    parse_row,
    verify,
)

from ._protocol import Hint


# Same time budget the bits provider effectively uses (search-bounded by
# the constrained family-F enumerator). Match the bits-pipeline budget so
# the two domains have comparable drop rates.
_TIME_BUDGET_SEC = 12.0


def _summarize_witness(w: dict) -> tuple[str, list[str]]:
    """Render the witness dict into (rule_summary, per_component).

    The witness shape from ``verifiers.equations.verify``:
      - ``stage``      — search-stage tag (e.g. ``"forward:template-only"``,
                          ``"mirror:q-op-only-arith"``)
      - ``templates``  — {op_char: rule_str} for each operator the search
                         pinned with a literal/positional template
      - ``arith``      — {op_char: family_name} for arithmetic families
                         (e.g. ``"a + b"``, ``"a XOR b mod 100"``)
      - ``bijection``  — {symbol: digit} if the row used the symbol→digit
                         convention (the "mixed" bucket)
    Arithmetic verifier (``method == "arith"``) returns a different shape
    via ``equations_arith.solve_row``; we render it best-effort.
    """
    parts: list[str] = []
    summary_bits: list[str] = []

    stage = w.get("stage", "")
    if stage:
        parts.append(f"stage: {stage}")

    templates = w.get("templates") or {}
    if templates:
        for op, rule in templates.items():
            parts.append(f"  op '{op}': {rule}")
        summary_bits.append(
            "templates: " + ", ".join(f"{op!r}→{r}" for op, r in templates.items())
        )

    arith = w.get("arith") or {}
    if arith:
        for op, fam in arith.items():
            parts.append(f"  op '{op}': arithmetic family — {fam}")
        summary_bits.append(
            "arith: " + ", ".join(f"{op!r}→{f}" for op, f in arith.items())
        )

    bij = w.get("bijection") or {}
    if bij:
        bij_str = ", ".join(f"{s}={d}" for s, d in sorted(bij.items()))
        parts.append(f"  symbol→digit bijection: {bij_str}")
        summary_bits.append(f"bijection({len(bij)} symbols)")

    rule_summary = "; ".join(summary_bits) if summary_bits else stage or "rule fits"
    return rule_summary, parts


def compute_hint(prompt: str, kaggle_answer: str) -> Hint | None:
    examples, q_lhs = parse_row(prompt)
    if not examples or not q_lhs:
        return None
    v = verify(prompt, kaggle_answer, time_budget_sec=_TIME_BUDGET_SEC)
    if v.get("status") != "ok":
        return None  # No rule in family F lands at stored answer — drop.

    witness = v.get("witness")
    if not isinstance(witness, dict):
        # Arithmetic-verifier path may stash a non-dict witness; treat as
        # a clean hit with a minimal summary.
        return Hint(
            rule_summary=f"verifier method: {v.get('method', 'unknown')}",
            per_component=[],
            applies_cleanly=True,
            extras={
                "examples": examples,
                "question": q_lhs,
                "kaggle_answer": kaggle_answer,
                "method": v.get("method", ""),
            },
        )

    rule_summary, per_component = _summarize_witness(witness)
    return Hint(
        rule_summary=rule_summary,
        per_component=per_component,
        applies_cleanly=True,
        extras={
            "examples": examples,
            "question": q_lhs,
            "kaggle_answer": kaggle_answer,
            "method": v.get("method", ""),
            "witness_raw": witness,
        },
    )


__all__ = ["compute_hint"]
