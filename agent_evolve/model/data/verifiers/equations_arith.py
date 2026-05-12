"""Consistency verifier for symbolic-equation labels (equations domain).

Format observed in balanced_dev600:

    <d><d><op><d><d> = <result-symbols>

where the middle char of LHS is one of ~2 operator symbols (same row uses
one or two distinct operators; different rows use different operator pairs
from the pool +, -, *, concat, absdiff, ...). The 2-char operand strings
encode 2-digit numbers under a row-specific bijection symbol → digit.

We verify labels by searching for:
  * a bijection digit_map: digit_symbol → 0..9
  * a mapping op_map: operator_symbol → arithmetic function
such that every example's arithmetic checks AND the question evaluates to
the stored answer.

Using z3-SMT for fast constraint propagation. Per-row wall is sub-second
on the typical 12-13 unique symbols.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/fsx/zzsamshi/a-evolve")

import z3


# Operator candidates: same set that covers Kaggle's equations generator.
# Each is (name, z3_body_builder(a,b) -> Int expression or list of Int expressions
# in case of concat where result size varies).
OPS_ARITH = {
    "+":          lambda a, b: a + b,
    "-":          lambda a, b: a - b,
    "revsub":     lambda a, b: b - a,
    "*":          lambda a, b: a * b,
    "absdiff":    lambda a, b: z3.If(a >= b, a - b, b - a),
    "negabsdiff": lambda a, b: z3.If(a >= b, b - a, a - b),
    "add+1":      lambda a, b: a + b + 1,
    "add-1":      lambda a, b: a + b - 1,
    "mul+1":      lambda a, b: a * b + 1,
    "mul-1":      lambda a, b: a * b - 1,
    "sub+1":      lambda a, b: a - b + 1,
    "sub-1":      lambda a, b: a - b - 1,
    # Digit-level ops (2-digit operands: a=10*a1+a0, b=10*b1+b0)
    # Kaggle's "equations" domain uses digit-wise operations too.
    # digit absolute difference per position, result as 2-digit number:
    # abs(a0 - b0) + 10 * abs(a1 - b1)
    # Express symbolically — works with a, b being z3 Ints (10..99 typically).
    # We implement using (a // 10), (a % 10), etc.
    "digit_add_mod10":
        lambda a, b: ((z3.IntVal(1) * ((a / 10) + (b / 10))) % 10) * 10
                     + (((a % 10) + (b % 10)) % 10),
    "digit_sub_mod10":
        lambda a, b: (((a / 10) - (b / 10)) % 10) * 10
                     + (((a % 10) - (b % 10)) % 10),
    "digit_absdiff":
        lambda a, b: (
            z3.If((a / 10) >= (b / 10), (a / 10) - (b / 10), (b / 10) - (a / 10)) * 10
            + z3.If((a % 10) >= (b % 10), (a % 10) - (b % 10), (b % 10) - (a % 10))
        ),
}
# "Concat" and "reverse concat" can't be expressed directly as an Int relation
# because result has different digit count than 2+2=4 base-10. We handle these
# in a separate pre-solve pass (deterministic: concat(a,b) gives a specific RHS
# pattern of length 4, rconcat length 4).


def _parse_row(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    """Pull pure-symbol `X = Y` examples and the `determine the result for:` Q."""
    examples: list[tuple[str, str]] = []
    q = None
    for line in prompt.splitlines():
        ln = line.strip()
        m = re.search(r"determine the result for:\s*(\S+)", ln)
        if m:
            q = m.group(1)
            continue
        m = re.match(r"^(\S+)\s*=\s*(\S+)\s*$", ln)
        if m and not re.search(r"[A-Za-z]", m.group(1) + m.group(2)):
            examples.append((m.group(1), m.group(2)))
    return examples, q


def _syms_to_int(s: str, digit_vars: dict[str, z3.ExprRef]) -> z3.ExprRef:
    """Build the z3 integer-value expression for a symbol string interpreted
    base-10 with symbol→digit bijection digit_vars. Assumes all chars in s
    are in digit_vars. Leading-zero constraint is added separately by caller.
    """
    val = z3.IntVal(0)
    for c in s:
        val = val * 10 + digit_vars[c]
    return val


def solve_row(examples: list[tuple[str, str]], q_lhs: str, stored_rhs: str,
              time_budget_sec: float = 5.0) -> tuple[bool, str | None]:
    """Search for a consistent digit+op mapping. Return (consistent, witness)."""
    if not examples or not q_lhs:
        return False, None

    # LHS length must be constant = 5 (op at pos 2). Relax to allow pos being
    # discoverable: use the position where symbols are sparsest.
    lhs_strings = [lhs for lhs, _ in examples] + [q_lhs]
    if len({len(s) for s in lhs_strings}) != 1:
        return False, None
    L = len(lhs_strings[0])
    # Find op_pos: the position with the fewest unique symbols.
    # Use only example LHS strings (not q).
    min_pos, min_n = 0, 99
    for p in range(L):
        n_uniq = len({lhs[p] for lhs, _ in examples} | {q_lhs[p]})
        if n_uniq < min_n:
            min_n = n_uniq
            min_pos = p

    op_pos = min_pos
    # Operator symbols across examples + question
    op_syms_by_ex = [lhs[op_pos] for lhs, _ in examples] + [q_lhs[op_pos]]
    unique_op_syms = sorted(set(op_syms_by_ex))

    # Decompose each example: (left_str, right_str, rhs_str, op_sym)
    decomposed = []
    for lhs, rhs in examples:
        decomposed.append((lhs[:op_pos], lhs[op_pos + 1:], rhs, lhs[op_pos]))
    q_left, q_right = q_lhs[:op_pos], q_lhs[op_pos + 1:]
    q_op_sym = q_lhs[op_pos]

    # Digit alphabet = union of chars appearing in operand + RHS positions
    # across examples + question + stored_rhs. An op symbol CAN also appear
    # as a digit in a different column (rare but possible in dev), so we
    # don't remove op syms here. If an op sym appears in a digit position,
    # it gets both a digit and an op interpretation; z3 can still solve.
    digit_syms: set[str] = set()
    for left, right, rhs, _ in decomposed:
        digit_syms |= set(left) | set(right) | set(rhs)
    digit_syms |= set(q_left) | set(q_right) | set(stored_rhs)
    # Only strip an op sym from digit alphabet if that sym NEVER appears in
    # operand/rhs positions (i.e. it's purely the operator).
    for op_s in unique_op_syms:
        if op_s in digit_syms:
            # Check if it's also in a digit position; if only in op positions, drop.
            in_digit_position = False
            for left, right, rhs, _ in decomposed:
                if op_s in left or op_s in right or op_s in rhs:
                    in_digit_position = True; break
            if not in_digit_position and op_s not in (q_left + q_right + stored_rhs):
                digit_syms.discard(op_s)
    if len(digit_syms) > 10:
        return False, None
    digit_syms = sorted(digit_syms)

    # For each assignment of op_syms → OPS_ARITH names, try z3 for bijection.
    from itertools import product
    deadline = time.time() + time_budget_sec
    for combo in product(OPS_ARITH.keys(), repeat=len(unique_op_syms)):
        if time.time() > deadline:
            return False, None
        op_assignment = dict(zip(unique_op_syms, combo))

        s = z3.Solver()
        s.set("timeout", max(50, int((deadline - time.time()) * 1000 / 4)))  # ms
        # Digit variables
        digit_vars = {sym: z3.Int(f"d_{i}") for i, sym in enumerate(digit_syms)}
        for v in digit_vars.values():
            s.add(v >= 0, v <= 9)
        # Bijection: all digits distinct
        s.add(z3.Distinct(*digit_vars.values()))
        # Example constraints
        feasible = True
        for left, right, rhs, op_sym in decomposed:
            if not left or not right or not rhs:
                feasible = False
                break
            op_fn = OPS_ARITH[op_assignment[op_sym]]
            a = _syms_to_int(left, digit_vars)
            b = _syms_to_int(right, digit_vars)
            computed = op_fn(a, b)
            # Leading-zero: first char of operand must be != 0 (unless operand has length 1)
            if len(left) > 1:
                s.add(digit_vars[left[0]] != 0)
            if len(right) > 1:
                s.add(digit_vars[right[0]] != 0)
            # RHS can be positive, zero, or (if it starts with a sign-marker) negative.
            # Represent: either the whole rhs decodes as a non-negative int and equals computed,
            #         or rhs[1:] decodes and rhs[0] is a sign-marker, and the value is -that.
            rhs_plain = _syms_to_int(rhs, digit_vars)
            # RHS can legitimately have a leading zero when the encoding left-pads
            # the computed value to a fixed length. So we don't force non-zero leading.
            positive_branch = rhs_plain == computed
            # Negative branch: rhs has len >= 2, rhs[0] is a sign marker, rhs[1:] decodes to -computed
            if len(rhs) >= 2:
                rhs_rest = _syms_to_int(rhs[1:], digit_vars)
                negative_branch = z3.And(computed < 0, rhs_rest == -computed)
                s.add(z3.Or(positive_branch, negative_branch))
            else:
                s.add(positive_branch)
        if not feasible:
            continue
        # Question constraint
        op_fn_q = OPS_ARITH[op_assignment[q_op_sym]]
        qa = _syms_to_int(q_left, digit_vars)
        qb = _syms_to_int(q_right, digit_vars)
        if len(q_left) > 1:
            s.add(digit_vars[q_left[0]] != 0)
        if len(q_right) > 1:
            s.add(digit_vars[q_right[0]] != 0)
        q_computed = op_fn_q(qa, qb)
        if stored_rhs:
            rhs_plain_q = _syms_to_int(stored_rhs, digit_vars)
            positive_branch_q = rhs_plain_q == q_computed
            if len(stored_rhs) >= 2:
                rhs_rest_q = _syms_to_int(stored_rhs[1:], digit_vars)
                negative_branch_q = z3.And(q_computed < 0, rhs_rest_q == -q_computed)
                s.add(z3.Or(positive_branch_q, negative_branch_q))
            else:
                s.add(positive_branch_q)

        r = s.check()
        if r == z3.sat:
            m = s.model()
            digit_map = {sym: m[v].as_long() for sym, v in digit_vars.items()}
            return True, f"op_pos={op_pos} ops={op_assignment} digits={digit_map}"
    # Try concat as separate pass: no SMT needed; deterministic decoding.
    for op_sym_concat in unique_op_syms:
        # concat(a,b): digits of a followed by digits of b.
        # Given examples of form "AB|CD = ABCD" where | is concat op, we can
        # directly read off digit_map: operand symbols map to "whatever digits
        # they are" and rhs is just the concatenation, so digit_map is
        # constrained by: rhs[0]=a[0], rhs[1]=a[1], rhs[2]=b[0], rhs[3]=b[1].
        # But that's a symbol-identity constraint (not a digit-equality).
        # Easier: check "do all examples using this op have rhs == left+right?"
        op_examples = [(l, r, rhs) for l, r, rhs, s_ in decomposed if s_ == op_sym_concat]
        if not op_examples:
            continue
        is_concat = all(rhs == left + right for left, right, rhs in op_examples)
        is_rconcat = all(rhs == right + left for left, right, rhs in op_examples)
        if not (is_concat or is_rconcat):
            continue
        # Great — op_sym_concat is concat or rconcat. But we still need arithmetic
        # consistency on other operator symbols. Skip this branch if other ops exist
        # and not yet handled — the SMT above already tried them.
        # For single-op rows where the only op IS concat:
        if len(unique_op_syms) == 1 and q_op_sym == op_sym_concat:
            # Question: concat or rconcat
            if is_concat:
                expected = q_left + q_right
            else:
                expected = q_right + q_left
            if expected == stored_rhs:
                return True, f"op_pos={op_pos} concat({op_sym_concat})"

    return False, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--time-budget", type=float, default=3.0)
    ap.add_argument("--domain", default="equations")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.input, newline="")) if r["domain"] == args.domain]
    print(f"checking {len(rows)} {args.domain} rows (budget {args.time_budget}s/row)")

    results: list[dict[str, Any]] = []
    n_ok = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        examples, q = _parse_row(r["prompt"])
        consistent, witness = False, None
        if examples and q:
            consistent, witness = solve_row(examples, q, r["answer"], args.time_budget)
        results.append({
            "id": r["id"], "source": r.get("source", ""),
            "stored_answer": r["answer"],
            "consistent": consistent,
            "witness": witness,
            "n_examples": len(examples),
        })
        if consistent:
            n_ok += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}  consistent={n_ok}  elapsed={time.time()-t0:.0f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for v in results:
            f.write(json.dumps(v) + "\n")

    print(f"\nconsistent:   {n_ok}  ({100*n_ok/len(rows):.1f}%)")
    print(f"inconsistent: {len(rows)-n_ok}  ({100*(len(rows)-n_ok)/len(rows):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
