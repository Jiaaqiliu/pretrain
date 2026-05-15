"""Verifier for the Kaggle ``cipher`` domain (substitution cipher with
optional Wonderland-word disambiguation).

The cipher key is recovered from the examples via constraint propagation
(see ``reasoning_cipher``). Verification is: run the default solver and
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
from agent_evolve.model.data.reasoners.cipher import solve
from agent_evolve.model.data.reasoners.store_types import Example, Problem

DOMAIN = "cipher"


def parse(prompt: str, stored_answer: str = "", _id: str = "") -> Problem | None:
    examples: list[Example] = []
    for ln in prompt.splitlines():
        m = re.match(r"^\s*([^-]+?)\s*->\s*(.+?)\s*$", ln)
        if not m:
            continue
        if "wonderland" in ln.lower() or "example" in ln.lower():
            continue
        left = m.group(1).strip()
        right = m.group(2).strip()
        if re.fullmatch(r"[a-z ]+", left) and re.fullmatch(r"[a-z ]+", right):
            examples.append(Example(left, right))
    q = re.search(r"decrypt the following text:\s*(.+?)\s*(\n|$)", prompt)
    if not examples or not q:
        return None
    return Problem(
        id=_id,
        category="cipher",
        examples=examples,
        question=q.group(1).strip(),
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
