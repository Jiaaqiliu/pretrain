"""Verifier for the Kaggle ``units`` domain (linear ``output = factor * input``).

The factor is recovered uniquely from any example with non-zero input, so
verification is: run the default ``reasoning_unit_conversion`` solver and
compare its ``\\boxed{}`` to the stored answer.

Exports:
    parse(prompt, stored_answer="", _id="") -> Problem | None
    verify(prompt, stored_answer)           -> dict
"""

from __future__ import annotations

import re
from typing import Any

from agent_evolve.benchmarks.nemo_reasoner import extract_final_answer
from agent_evolve.benchmarks.nemo_reasoner import verify as _kaggle_verify
from agent_evolve.model.data.reasoners.store_types import Example, Problem
from agent_evolve.model.data.reasoners.units import solve

DOMAIN = "units"


def parse(prompt: str, stored_answer: str = "", _id: str = "") -> Problem | None:
    pairs = re.findall(r"([-+]?\d*\.?\d+)\s*m\s+becomes\s+([-+]?\d*\.?\d+)", prompt)
    q = re.search(r"convert the following measurement:\s*([-+]?\d*\.?\d+)\s*m", prompt)
    if not pairs or not q:
        return None
    return Problem(
        id=_id,
        category="unit_conversion",
        examples=[Example(i, o) for i, o in pairs],
        question=q.group(1),
        answer=stored_answer,
        prompt=prompt,
    )


def verify(prompt: str, stored_answer: str) -> dict[str, Any]:
    base = {"domain": DOMAIN, "agrees": False, "prediction": None,
            "status": "", "witness": None}
    problem = parse(prompt, stored_answer)
    if problem is None:
        return {**base, "status": "parse_failed"}
    try:
        trace = solve(problem)
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": f"solver_error: {exc!r}"}
    if trace is None:
        return {**base, "status": "no_solution"}
    pred = extract_final_answer(trace)
    if not pred:
        return {**base, "prediction": pred, "status": "no_boxed"}
    return {**base, "prediction": pred, "status": "ok",
            "agrees": bool(_kaggle_verify(stored_answer, pred))}
