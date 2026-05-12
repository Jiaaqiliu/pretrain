"""Programmatic label verification for the balanced_dev600 dev set.

Uses huikang's open-sourced domain solvers (ported into
agent_evolve.model.data.reasoners) as ground-truth oracles for all 6
Kaggle-scored domains. For each dev row we:

  1. Parse the plain-text Kaggle prompt into a ``Problem`` (examples +
     question + stored answer).
  2. Run the category's ``reasoning_<domain>`` function.
  3. Extract ``\\boxed{..}`` from the reasoning trace and compare to the
     stored answer with the same ``verify()`` the dev scorer uses.

Disagreement indicates either a label error in the dev set OR a solver
that couldn't handle a particular example edge case.

Input:  balanced_dev600.csv with (id, prompt, answer, domain, source).
Output: balanced_dev600.programmatic.jsonl with per-row verdicts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, "/fsx/zzsamshi/a-evolve")

from agent_evolve.benchmarks.nemo_reasoner import extract_final_answer, verify  # noqa: E402
from agent_evolve.model.data.reasoners.bit_manipulation import reasoning_bit_manipulation  # noqa: E402
from agent_evolve.model.data.reasoners.cipher import reasoning_cipher  # noqa: E402
from agent_evolve.model.data.reasoners.cryptarithm import reasoning_cryptarithm  # noqa: E402
from agent_evolve.model.data.reasoners.equation_numeric import reasoning_equation_numeric  # noqa: E402
from agent_evolve.model.data.reasoners.gravity import reasoning_gravity  # noqa: E402
from agent_evolve.model.data.reasoners.numeral import reasoning_numeral  # noqa: E402
from agent_evolve.model.data.reasoners.store_types import Example, Problem  # noqa: E402
from agent_evolve.model.data.reasoners.unit_conversion import reasoning_unit_conversion  # noqa: E402

logger = logging.getLogger(__name__)


# ── Prompt parsers: plain text → Problem ───────────────────────────────────


def _parse_bits(prompt: str, answer: str, _id: str) -> Problem | None:
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    q = re.search(r"determine the output for:\s*([01]{8})", prompt)
    if not pairs or not q:
        return None
    examples = [Example(i, o) for i, o in pairs]
    return Problem(id=_id, category="bit_manipulation", examples=examples,
                   question=q.group(1), answer=answer, prompt=prompt)


def _parse_numerals(prompt: str, answer: str, _id: str) -> Problem | None:
    # Examples like "23 -> XXIII"
    pairs = re.findall(r"(\d+)\s*->\s*([IVXLCDM]+)", prompt)
    q = re.search(r"write the number (\d+) in the Wonderland", prompt)
    if not pairs or not q:
        return None
    examples = [Example(i, o) for i, o in pairs]
    return Problem(id=_id, category="numeral", examples=examples,
                   question=q.group(1), answer=answer, prompt=prompt)


def _parse_units(prompt: str, answer: str, _id: str) -> Problem | None:
    # "24.12 m becomes 34.78"
    pairs = re.findall(r"([-+]?\d*\.?\d+)\s*m\s+becomes\s+([-+]?\d*\.?\d+)", prompt)
    q = re.search(r"convert the following measurement:\s*([-+]?\d*\.?\d+)\s*m", prompt)
    if not pairs or not q:
        return None
    examples = [Example(i, o) for i, o in pairs]
    return Problem(id=_id, category="unit_conversion", examples=examples,
                   question=q.group(1), answer=answer, prompt=prompt)


def _parse_gravity(prompt: str, answer: str, _id: str) -> Problem | None:
    # "For t = 1.54s, distance = 11.74 m"
    pairs = re.findall(
        r"For t\s*=\s*([-+]?\d*\.?\d+)s?,\s*distance\s*=\s*([-+]?\d*\.?\d+)\s*m",
        prompt,
    )
    q = re.search(r"falling distance for t\s*=\s*([-+]?\d*\.?\d+)s?", prompt)
    if not pairs or not q:
        return None
    examples = [Example(t, d) for t, d in pairs]
    return Problem(id=_id, category="gravity", examples=examples,
                   question=q.group(1), answer=answer, prompt=prompt)


def _parse_cipher(prompt: str, answer: str, _id: str) -> Problem | None:
    # Examples: "<cipher words> -> <plaintext words>"
    # Take every "-> " line; question is after "decrypt the following text:"
    lines = prompt.splitlines()
    examples = []
    for ln in lines:
        m = re.match(r"^\s*([^-]+?)\s*->\s*(.+?)\s*$", ln)
        if m and "wonderland" not in ln.lower() and "example" not in ln.lower():
            left = m.group(1).strip()
            right = m.group(2).strip()
            # Only plausible cipher examples: alphabetic with spaces
            if re.fullmatch(r"[a-z ]+", left) and re.fullmatch(r"[a-z ]+", right):
                examples.append(Example(left, right))
    q = re.search(r"decrypt the following text:\s*(.+?)\s*(\n|$)", prompt)
    if not examples or not q:
        return None
    return Problem(id=_id, category="cipher", examples=examples,
                   question=q.group(1).strip(), answer=answer, prompt=prompt)


def _parse_equations(prompt: str, answer: str, _id: str) -> Problem | None:
    # "X = Y" lines where X and Y are opaque symbol strings
    # and question "determine the result for: X"
    lines = prompt.splitlines()
    examples = []
    for ln in lines:
        m = re.match(r"^\s*(\S+)\s*=\s*(\S+)\s*$", ln)
        if m:
            examples.append(Example(m.group(1), m.group(2)))
    q = re.search(r"determine the result for:\s*(\S+)", prompt)
    if not examples or not q:
        return None
    return Problem(id=_id, category="equation_numeric_deduce", examples=examples,
                   question=q.group(1), answer=answer, prompt=prompt)


# Router: dev-set domain → (parser, solver)
SOLVERS: dict[str, tuple[Callable, Callable]] = {
    "bits":      (_parse_bits,      reasoning_bit_manipulation),
    "numerals":  (_parse_numerals,  reasoning_numeral),
    "units":     (_parse_units,     reasoning_unit_conversion),
    "gravity":   (_parse_gravity,   reasoning_gravity),
    "cipher":    (_parse_cipher,    reasoning_cipher),
    "equations": (_parse_equations, reasoning_equation_numeric),
}


# ── Runner ─────────────────────────────────────────────────────────────────


@dataclass
class Row:
    id: str
    prompt: str
    answer: str
    domain: str
    source: str = ""


def load_rows(path: Path, limit: int | None = None) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(Row(r["id"], r["prompt"], r["answer"], r.get("domain", ""), r.get("source", "")))
            if limit and len(rows) >= limit:
                break
    return rows


def judge_row(r: Row) -> dict:
    spec = SOLVERS.get(r.domain)
    base = {
        "id": r.id, "domain": r.domain, "source": r.source,
        "stored_answer": r.answer,
    }
    if spec is None:
        return {**base, "solver_prediction": None, "solver_status": "no_solver", "agrees": None}
    parser, solver = spec
    try:
        problem = parser(r.prompt, r.answer, r.id)
    except Exception as exc:  # noqa: BLE001
        return {**base, "solver_prediction": None, "solver_status": f"parse_error: {exc!r}", "agrees": None}
    if problem is None:
        return {**base, "solver_prediction": None, "solver_status": "parse_failed", "agrees": None}
    try:
        reasoning = solver(problem)
    except Exception as exc:  # noqa: BLE001
        return {**base, "solver_prediction": None, "solver_status": f"solver_error: {exc!r}", "agrees": None}
    if reasoning is None:
        return {**base, "solver_prediction": None, "solver_status": "no_solution", "agrees": None}
    pred = extract_final_answer(reasoning)
    if pred is None or not pred:
        return {**base, "solver_prediction": pred, "solver_status": "no_boxed", "agrees": None}
    agrees = verify(r.answer, pred)
    return {**base, "solver_prediction": pred, "solver_status": "ok", "agrees": agrees}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_rows(args.input, limit=args.limit)
    logger.info("loaded %d rows from %s", len(rows), args.input)

    verdicts = [judge_row(r) for r in rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")

    # Summary
    from collections import defaultdict
    per = defaultdict(lambda: {"n": 0, "ok": 0, "agree": 0, "disagree": 0, "parse_fail": 0, "no_solution": 0, "no_boxed": 0, "errors": 0})
    for v in verdicts:
        b = per[v["domain"]]
        b["n"] += 1
        s = v["solver_status"]
        if s == "ok":
            b["ok"] += 1
            if v["agrees"]:
                b["agree"] += 1
            else:
                b["disagree"] += 1
        elif s == "parse_failed":
            b["parse_fail"] += 1
        elif s == "no_solution":
            b["no_solution"] += 1
        elif s == "no_boxed":
            b["no_boxed"] += 1
        else:
            b["errors"] += 1

    print("\n=== programmatic label verification (huikang solvers) ===")
    print(f"{'domain':10s} {'n':>4} {'solved':>7} {'agree':>6} {'disagree':>9} "
          f"{'parse_fail':>11} {'no_sol':>7} {'no_box':>7} {'errors':>7}  "
          f"{'label_trust':>12}")
    for d in sorted(per):
        b = per[d]
        rate = 100 * b["agree"] / max(1, b["ok"])
        print(f"{d:10s} {b['n']:>4d} {b['ok']:>7d} {b['agree']:>6d} {b['disagree']:>9d} "
              f"{b['parse_fail']:>11d} {b['no_solution']:>7d} {b['no_boxed']:>7d} {b['errors']:>7d}  "
              f"{rate:>10.1f}%")

    bad = [v for v in verdicts if v["agrees"] is False]
    if bad:
        print(f"\n=== {len(bad)} disagreements (labels worth human review) ===")
        for v in bad[:20]:
            print(f"  id={v['id']} dom={v['domain']:10s} src={v['source']:13s} "
                  f"stored={v['stored_answer']!r}  solver={v['solver_prediction']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
