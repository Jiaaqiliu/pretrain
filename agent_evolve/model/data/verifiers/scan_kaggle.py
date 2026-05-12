"""Scan the full Kaggle ``train.csv`` (9,500 rows, ``id,prompt,answer``) with
every verifier we have, recording per-row verdicts + per-domain coverage.

For each row:

  1. Infer domain from prompt text (no ``domain`` column in Kaggle CSV).
  2. Dispatch to the strongest verifier we have for that domain:
     * numerals / units / gravity / cipher → huikang ``reasoning_*`` solver
       (deterministic; if the boxed answer matches the stored answer the row
       is "solver-agrees").
     * bits → ``is_bits_label_consistent`` (≤4-input truth-table +
       global byte transforms); "agrees" = label is example-consistent.
     * equations → first the arithmetic z3 verifier; if that says inconsistent,
       fall back to the S1–S5 symbolic verifier. "agrees" = either says
       consistent.

Output: ``kaggle_train.scan.jsonl`` with one row per input + summary printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, "/fsx/zzsamshi/a-evolve")

from agent_evolve.benchmarks.nemo_reasoner import extract_final_answer, verify  # noqa: E402

from agent_evolve.model.data.verifiers.bits import is_bits_label_consistent  # noqa: E402
from agent_evolve.model.data.verifiers.equations_arith import solve_row as solve_eq_arith  # noqa: E402
from agent_evolve.model.data.verifiers.equations_arith import _parse_row as parse_eq_row  # noqa: E402
from agent_evolve.model.data.verifiers.equations import solve_row as solve_eq_symbolic  # noqa: E402
from agent_evolve.model.data.verifiers.equations import parse_row as parse_eq_sym_row  # noqa: E402
from agent_evolve.model.data.reasoners.bit_manipulation import reasoning_bit_manipulation  # noqa: E402
from agent_evolve.model.data.reasoners.cipher import reasoning_cipher  # noqa: E402
from agent_evolve.model.data.reasoners.gravity import reasoning_gravity  # noqa: E402
from agent_evolve.model.data.reasoners.numeral import reasoning_numeral  # noqa: E402
from agent_evolve.model.data.reasoners.store_types import Example, Problem  # noqa: E402
from agent_evolve.model.data.reasoners.unit_conversion import reasoning_unit_conversion  # noqa: E402


# ── Domain inference ─────────────────────────────────────────────────────

def infer_domain(prompt: str) -> str:
    p = prompt
    if "8-bit binary" in p or "determine the output for:" in p:
        return "bits"
    if "transformation rules is applied to equations" in p or "determine the result for:" in p:
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


# ── Parsers for each domain ──────────────────────────────────────────────

def _parse_bits_prompt(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    q = re.search(r"determine the output for:\s*([01]{8})", prompt)
    return pairs, (q.group(1) if q else None)


def _problem_numerals(prompt: str, answer: str, _id: str) -> Problem | None:
    pairs = re.findall(r"(\d+)\s*->\s*([IVXLCDM]+)", prompt)
    q = re.search(r"write the number (\d+) in the Wonderland", prompt)
    if not pairs or not q:
        return None
    examples = [Example(i, o) for i, o in pairs]
    return Problem(id=_id, category="numeral", examples=examples,
                   question=q.group(1), answer=answer, prompt=prompt)


def _problem_units(prompt: str, answer: str, _id: str) -> Problem | None:
    pairs = re.findall(r"([-+]?\d*\.?\d+)\s*m\s+becomes\s+([-+]?\d*\.?\d+)", prompt)
    q = re.search(r"convert the following measurement:\s*([-+]?\d*\.?\d+)\s*m", prompt)
    if not pairs or not q:
        return None
    examples = [Example(i, o) for i, o in pairs]
    return Problem(id=_id, category="unit_conversion", examples=examples,
                   question=q.group(1), answer=answer, prompt=prompt)


def _problem_gravity(prompt: str, answer: str, _id: str) -> Problem | None:
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


def _problem_cipher(prompt: str, answer: str, _id: str) -> Problem | None:
    lines = prompt.splitlines()
    examples = []
    for ln in lines:
        m = re.match(r"^\s*([^-]+?)\s*->\s*(.+?)\s*$", ln)
        if m and "wonderland" not in ln.lower() and "example" not in ln.lower():
            left = m.group(1).strip()
            right = m.group(2).strip()
            if re.fullmatch(r"[a-z ]+", left) and re.fullmatch(r"[a-z ]+", right):
                examples.append(Example(left, right))
    q = re.search(r"decrypt the following text:\s*(.+?)\s*(\n|$)", prompt)
    if not examples or not q:
        return None
    return Problem(id=_id, category="cipher", examples=examples,
                   question=q.group(1).strip(), answer=answer, prompt=prompt)


# ── Per-domain verifier wrappers ─────────────────────────────────────────

def verify_programmatic(solver, problem: Problem, stored_answer: str) -> dict[str, Any]:
    try:
        reasoning = solver(problem)
    except Exception as exc:  # noqa: BLE001
        return {"method": "huikang", "status": f"solver_error: {exc!r}", "agrees": False}
    if reasoning is None:
        return {"method": "huikang", "status": "no_solution", "agrees": False}
    pred = extract_final_answer(reasoning)
    if not pred:
        return {"method": "huikang", "status": "no_boxed", "agrees": False}
    return {"method": "huikang", "status": "ok",
            "agrees": bool(verify(stored_answer, pred)),
            "prediction": pred}


def verify_bits(prompt: str, answer: str) -> dict[str, Any]:
    pairs, q = _parse_bits_prompt(prompt)
    if not pairs or not q:
        return {"method": "bits_consistency", "status": "parse_failed", "agrees": False}
    try:
        consistent, witnesses = is_bits_label_consistent(pairs, q, answer)
    except Exception as exc:  # noqa: BLE001
        return {"method": "bits_consistency", "status": f"error: {exc!r}", "agrees": False}
    return {"method": "bits_consistency",
            "status": "ok",
            "agrees": bool(consistent),
            "witness_rules": witnesses if consistent else None}


def verify_equations(prompt: str, answer: str, time_budget: float) -> dict[str, Any]:
    # Pass 1: arithmetic z3 (fast when it succeeds).
    examples, q_lhs = parse_eq_row(prompt)
    if not examples or not q_lhs:
        return {"method": "equations", "status": "parse_failed", "agrees": False}
    try:
        arith_ok, arith_witness = solve_eq_arith(examples, q_lhs, answer,
                                                  time_budget_sec=time_budget * 0.4)
    except Exception as exc:  # noqa: BLE001
        arith_ok, arith_witness = False, f"error: {exc!r}"
    if arith_ok:
        return {"method": "eq_arith", "status": "ok", "agrees": True,
                "witness": arith_witness}
    # Pass 2: S1–S5 symbolic solver.
    ex2, q_lhs2 = parse_eq_sym_row(prompt)
    if not ex2 or not q_lhs2:
        return {"method": "eq_symbolic", "status": "parse_failed", "agrees": False}
    try:
        sym_ok, sym_witness = solve_eq_symbolic(ex2, q_lhs2, answer,
                                                 time_budget_sec=time_budget)
    except Exception as exc:  # noqa: BLE001
        return {"method": "eq_symbolic", "status": f"error: {exc!r}", "agrees": False}
    if sym_ok:
        return {"method": "eq_symbolic", "status": "ok", "agrees": True,
                "witness": sym_witness}
    return {"method": "eq_none", "status": "unexplained", "agrees": False}


# ── Row dispatcher ───────────────────────────────────────────────────────

def scan_row(row: dict, eq_time_budget: float) -> dict[str, Any]:
    rid = row["id"]
    prompt = row["prompt"]
    answer = row["answer"]
    domain = infer_domain(prompt)
    base = {"id": rid, "domain": domain, "stored_answer": answer}
    if domain == "numerals":
        p = _problem_numerals(prompt, answer, rid)
        if p is None:
            return {**base, "method": "none", "status": "parse_failed", "agrees": False}
        return {**base, **verify_programmatic(reasoning_numeral, p, answer)}
    if domain == "units":
        p = _problem_units(prompt, answer, rid)
        if p is None:
            return {**base, "method": "none", "status": "parse_failed", "agrees": False}
        return {**base, **verify_programmatic(reasoning_unit_conversion, p, answer)}
    if domain == "gravity":
        p = _problem_gravity(prompt, answer, rid)
        if p is None:
            return {**base, "method": "none", "status": "parse_failed", "agrees": False}
        return {**base, **verify_programmatic(reasoning_gravity, p, answer)}
    if domain == "cipher":
        p = _problem_cipher(prompt, answer, rid)
        if p is None:
            return {**base, "method": "none", "status": "parse_failed", "agrees": False}
        return {**base, **verify_programmatic(reasoning_cipher, p, answer)}
    if domain == "bits":
        return {**base, **verify_bits(prompt, answer)}
    if domain == "equations":
        return {**base, **verify_equations(prompt, answer, eq_time_budget)}
    return {**base, "method": "none", "status": "unknown_domain", "agrees": False}


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="/tmp/kaggle_comp/train.csv", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eq-time-budget", type=float, default=3.0,
                    help="per-row seconds for equations solvers (arith + symbolic)")
    ap.add_argument("--domain", default=None, help="restrict to one domain")
    ap.add_argument("--shard", type=int, default=None,
                    help="shard index (0-based) — takes shard/num-shards slice")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input, newline="")))
    if args.domain:
        rows = [r for r in rows if infer_domain(r["prompt"]) == args.domain]
    if args.shard is not None and args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    print(f"scanning {len(rows)} rows  (equations budget {args.eq_time_budget}s/row)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    per_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    last_flush = t0
    with open(args.output, "w") as f:
        for i, r in enumerate(rows):
            v = scan_row(r, args.eq_time_budget)
            f.write(json.dumps(v) + "\n")
            d = v.get("domain", "?")
            per_domain[d]["n"] += 1
            if v.get("agrees"):
                per_domain[d]["ok"] += 1
            if (i + 1) % 100 == 0 or (time.time() - last_flush) > 60:
                f.flush()
                last_flush = time.time()
                print(f"  {i+1}/{len(rows)}  elapsed={time.time()-t0:.0f}s  "
                      + "  ".join(f"{k}:{v['ok']}/{v['n']}" for k, v in sorted(per_domain.items())))

    print("\n=== per-domain coverage ===")
    print(f"{'domain':10s} {'n':>6s} {'verified':>9s} {'%':>6s}")
    for d in sorted(per_domain):
        v = per_domain[d]
        pct = 100 * v["ok"] / max(1, v["n"])
        print(f"{d:10s} {v['n']:>6d} {v['ok']:>9d} {pct:>5.1f}%")
    total_n = sum(v["n"] for v in per_domain.values())
    total_ok = sum(v["ok"] for v in per_domain.values())
    print(f"{'total':10s} {total_n:>6d} {total_ok:>9d} {100*total_ok/max(1,total_n):>5.1f}%")
    print(f"elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
