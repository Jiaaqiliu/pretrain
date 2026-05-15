"""Per-domain label verifiers for the 6 Kaggle Nemotron-Reasoning domains.

Each domain has:

* a *solver* under :mod:`agent_evolve.model.data.reasoners` that produces a
  CoT trace ending in ``\\boxed{answer}`` (see ``reasoners.SOLVERS``).
* a *verifier* in this package — one file per Kaggle domain, each exposing
  the same uniform shape::

      parse(prompt, stored_answer="", _id="") -> Problem | None
      verify(prompt, stored_answer)           -> dict

  The ``verify`` return shape is uniform across domains::

      {"domain": str,
       "agrees": bool,
       "prediction": str | None,     # for well-determined domains: solver's
                                     #   boxed answer. For bits/equations:
                                     #   stored_answer when consistent.
       "status": "ok" | "parse_failed" | "no_solution" | "no_boxed" |
                 "solver_error: ..." | "verifier_error: ..." |
                 "unexplained",
       "witness": str | list | None}  # rule that explained the label
                                       # (bits, equations only)

  Well-determined domains (``cipher``, ``gravity``, ``numerals``, ``units``)
  run the default solver and compare its boxed answer to the stored label.

  Under-determined domains (``bits``, ``equations``) run the stricter
  family-membership check (does there exist *some* rule in the family that
  fits the examples *and* gives the stored answer for the question).

Modules:

* :mod:`.bits`      — bits verifier (8-bit rule-family consistency)
* :mod:`.cipher`    — cipher verifier (solver-as-oracle)
* :mod:`.equations` — equations verifier (cascading arith z3 → S1–S5 symbolic)
* :mod:`.equations_arith` — arithmetic z3 implementation (sub-module of equations)
* :mod:`.gravity`   — gravity verifier (solver-as-oracle)
* :mod:`.numerals`  — numerals verifier (solver-as-oracle)
* :mod:`.units`     — units verifier (solver-as-oracle)

Cross-domain tools:

* :mod:`.programmatic` — dataset-level scan with a `domain` column; uses
  the default solver as oracle. Used for dev-set audits.
* :mod:`.scan_kaggle` — dataset-level scan over the raw Kaggle
  ``train.csv`` (no domain column); infers domain per row.
* :mod:`.opus_judge` — independent LLM cross-check (Claude Opus 4.6 via
  Bedrock).

This module exposes a top-level :data:`VERIFIERS` map keyed by Kaggle
domain name, :func:`infer_domain` for prompt → domain inference, and a
convenience :func:`verify` that dispatches based on the inferred domain.

The :mod:`equations` verifier needs ``z3-solver``; if it is unavailable
the equations entry in :data:`VERIFIERS` is set to ``None`` and the other
five domains keep working.
"""

from __future__ import annotations

from typing import Any, Callable

from . import bits, cipher, gravity, numerals, units

_EQ_IMPORT_ERROR: Exception | None = None
try:
    from . import equations as _equations  # may need z3
except ImportError as _exc:  # pragma: no cover - exercised only when z3 missing
    _equations = None  # type: ignore[assignment]
    _EQ_IMPORT_ERROR = _exc

VerifyFn = Callable[[str, str], dict[str, Any]]

VERIFIERS: dict[str, VerifyFn | None] = {
    "bits":      bits.verify,
    "cipher":    cipher.verify,
    "equations": _equations.verify if _equations is not None else None,
    "gravity":   gravity.verify,
    "numerals":  numerals.verify,
    "units":     units.verify,
}

DOMAINS = tuple(VERIFIERS.keys())


def infer_domain(prompt: str) -> str:
    """Infer the Kaggle domain from raw prompt text.

    Returns one of ``DOMAINS`` or ``"unknown"``.
    """
    p = prompt
    if "8-bit binary" in p or "determine the output for:" in p:
        return "bits"
    if ("transformation rules is applied to equations" in p
            or "determine the result for:" in p):
        return "equations"
    if "falling distance" in p or ("For t =" in p and "distance" in p):
        return "gravity"
    if "write the number" in p and "Wonderland" in p:
        return "numerals"
    if "convert the following measurement" in p:
        return "units"
    if "decrypt the following text" in p:
        return "cipher"
    return "unknown"


def get_verifier(domain: str) -> VerifyFn | None:
    """Return the verifier for a Kaggle domain (or None if unknown / unavailable)."""
    return VERIFIERS.get(domain)


def verify(prompt: str, stored_answer: str) -> dict[str, Any]:
    """Top-level dispatch: infer the domain, then run that domain's verifier."""
    dom = infer_domain(prompt)
    fn = VERIFIERS.get(dom)
    if fn is None:
        if dom in VERIFIERS:
            return {"domain": dom, "agrees": False, "prediction": None,
                    "status": "verifier_unavailable", "witness": None,
                    "error": repr(_EQ_IMPORT_ERROR) if dom == "equations" else None}
        return {"domain": "unknown", "agrees": False, "prediction": None,
                "status": "unknown_domain", "witness": None}
    return fn(prompt, stored_answer)


__all__ = [
    "VERIFIERS",
    "DOMAINS",
    "infer_domain",
    "get_verifier",
    "verify",
    "bits",
    "cipher",
    "gravity",
    "numerals",
    "units",
]
if _equations is not None:
    __all__.append("equations")
