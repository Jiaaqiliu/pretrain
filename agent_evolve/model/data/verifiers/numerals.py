"""Verifier for the Kaggle ``numerals`` domain (Arabic → Roman).

Numerals are uniquely determined by the question integer, so verification
is simply: run the default ``reasoning_numeral`` solver and compare its
``\\boxed{}`` to the stored answer.

Exports:
    parse(prompt, stored_answer="", _id="") -> Problem | None
    verify(prompt, stored_answer)           -> dict
"""

from __future__ import annotations

import re
from typing import Any

from agent_evolve.benchmarks.nemo_reasoner import extract_final_answer
from agent_evolve.benchmarks.nemo_reasoner import verify as _kaggle_verify
from agent_evolve.model.data.reasoners.numerals import solve
from agent_evolve.model.data.reasoners.store_types import Example, Problem

DOMAIN = "numerals"


def parse(prompt: str, stored_answer: str = "", _id: str = "") -> Problem | None:
    pairs = re.findall(r"(\d+)\s*->\s*([IVXLCDM]+)", prompt)
    q = re.search(r"write the number (\d+) in the Wonderland", prompt)
    if not pairs or not q:
        return None
    return Problem(
        id=_id,
        category="numeral",
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
