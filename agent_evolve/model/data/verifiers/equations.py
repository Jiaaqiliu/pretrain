"""Symbolic-equation consistency verifier (S1–S5 rule families).

The equations domain is a mix of arithmetic and non-arithmetic rewrites. The
previous verifier (``equations_label_consistency.py``) only handled the
arithmetic case — a symbol→digit bijection plus one of ~15 arithmetic op
families — and declared every other row "inconsistent". That left 92/100
equation rows in ``balanced_dev600`` unverified.

This module extends the rule vocabulary to cover the five rule families the
Kaggle generator actually uses, composed independently per operator symbol
in the same row:

  S1. **String templates.** RHS is a fixed pick-and-concat over LHS positions
      (and/or literal chars). Covers ``concat`` / ``rconcat`` / ``drop-middle``
      / ``swap-halves`` / ``select-left`` / ``select-right`` / ``mirror`` etc.
  S2. **Literal-constant templates.** Same, but with a specific literal char
      pinned at one or more RHS positions. Catches ``punctuation rewrite``.
  S3. **Arithmetic with shared row bijection.** Same as the legacy solver
      but the bijection only needs to cover the chars that participate in
      numeric positions — template-only chars can be excluded.
  S4. **Mirror preprocessor.** Reverse the whole row (lhs and rhs) first,
      then apply any of S1–S3 rules.
  S5. **Mixed per-op.** A row can contain ops of different kinds: e.g. one op
      is concat (template), another is multiplication (arithmetic) — solved
      jointly with the row's shared digit bijection.

Returns ``(consistent, witness_dict)`` per row. ``witness_dict`` records, for
each op symbol, which rule explained its examples and (if arithmetic) the
bijection used.

Limitations:
  * We assume LHS length 5 with the operator at position 2 — matches every
    row in balanced_dev600 (verified) and the huikang generator.
  * We stop at the first satisfying rule set; we do not enumerate all.
  * Template families enumerate up to 6^L candidates at RHS length L; caps at
    L=5 (7,776 templates per op) so per-row wall stays sub-second.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, "/fsx/zzsamshi/a-evolve")

import z3

LHS_LEN = 5
OP_POS = 2
LHS_POSITIONS = (0, 1, 3, 4)  # operand positions (non-op)

# ── Arithmetic op families ────────────────────────────────────────────────
# (name, z3_body_builder)  — builder returns an Int z3 expression.
# The wider families from equation_numeric.py are covered through
# digit-level builders + a shared 2-digit-operand assumption.

def _z3_digit(a: z3.ExprRef, which: str) -> z3.ExprRef:
    """Extract tens (``t``) or units (``u``) digit of a 2-digit z3 Int."""
    return (a / 10) if which == "t" else (a % 10)


def _op_builders() -> dict[str, Callable[[z3.ExprRef, z3.ExprRef], z3.ExprRef]]:
    ops: dict[str, Callable[[z3.ExprRef, z3.ExprRef], z3.ExprRef]] = {
        "add":        lambda a, b: a + b,
        "sub":        lambda a, b: a - b,
        "revsub":     lambda a, b: b - a,
        "mul":        lambda a, b: a * b,
        "absdiff":    lambda a, b: z3.If(a >= b, a - b, b - a),
        "negabsdiff": lambda a, b: z3.If(a >= b, b - a, a - b),
        "add+1":      lambda a, b: a + b + 1,
        "add-1":      lambda a, b: a + b - 1,
        "mul+1":      lambda a, b: a * b + 1,
        "mul-1":      lambda a, b: a * b - 1,
        "sub+1":      lambda a, b: a - b + 1,
        "sub-1":      lambda a, b: a - b - 1,
    }
    # digit-level two-digit ops
    ops["digit_add_mod10"] = lambda a, b: (
        ((_z3_digit(a, "t") + _z3_digit(b, "t")) % 10) * 10
        + ((_z3_digit(a, "u") + _z3_digit(b, "u")) % 10)
    )
    ops["digit_sub_mod10"] = lambda a, b: (
        ((_z3_digit(a, "t") - _z3_digit(b, "t")) % 10) * 10
        + ((_z3_digit(a, "u") - _z3_digit(b, "u")) % 10)
    )
    ops["digit_absdiff"] = lambda a, b: (
        z3.If(_z3_digit(a, "t") >= _z3_digit(b, "t"),
              _z3_digit(a, "t") - _z3_digit(b, "t"),
              _z3_digit(b, "t") - _z3_digit(a, "t")) * 10
        + z3.If(_z3_digit(a, "u") >= _z3_digit(b, "u"),
                _z3_digit(a, "u") - _z3_digit(b, "u"),
                _z3_digit(b, "u") - _z3_digit(a, "u"))
    )
    ops["cross_mul"] = lambda a, b: (
        _z3_digit(a, "t") * _z3_digit(b, "t")
        + _z3_digit(a, "u") * _z3_digit(b, "u")
    )
    ops["cross_mul_rev"] = lambda a, b: (
        _z3_digit(a, "t") * _z3_digit(b, "u")
        + _z3_digit(a, "u") * _z3_digit(b, "t")
    )
    ops["determinant"] = lambda a, b: (
        _z3_digit(a, "t") * _z3_digit(b, "u")
        - _z3_digit(a, "u") * _z3_digit(b, "t")
    )
    ops["abs_determinant"] = lambda a, b: z3.Abs(
        _z3_digit(a, "t") * _z3_digit(b, "u")
        - _z3_digit(a, "u") * _z3_digit(b, "t")
    )
    # Additional digit-level ops from huikang's equation_numeric._rare_candidates
    ops["digit_mul"] = lambda a, b: (
        _z3_digit(a, "t") * _z3_digit(b, "t") * 10
        + _z3_digit(a, "u") * _z3_digit(b, "u")
    )
    ops["digit_mul_rev"] = lambda a, b: (
        _z3_digit(a, "t") * _z3_digit(b, "u") * 10
        + _z3_digit(a, "u") * _z3_digit(b, "t")
    )
    ops["digit_sum_diff"] = lambda a, b: (
        (_z3_digit(a, "t") + _z3_digit(a, "u"))
        - (_z3_digit(b, "t") + _z3_digit(b, "u"))
    )
    ops["digit_sum_sum"] = lambda a, b: (
        _z3_digit(a, "t") + _z3_digit(a, "u")
        + _z3_digit(b, "t") + _z3_digit(b, "u")
    )
    ops["digit_product_diff"] = lambda a, b: (
        _z3_digit(a, "t") * _z3_digit(a, "u")
        - _z3_digit(b, "t") * _z3_digit(b, "u")
    )
    ops["digit_product_sum"] = lambda a, b: (
        _z3_digit(a, "t") * _z3_digit(a, "u")
        + _z3_digit(b, "t") * _z3_digit(b, "u")
    )
    # 9's-complement subtraction (2-digit operands): sign(a-b) * (a + (99 - b))
    # Kaggle example: 16-71 = -44  → 16 + (99-71) = 44, sign negative since 16<71.
    # The RHS magnitude is the sum a + 99 - b; we add this as an op and the caller's
    # signed-RHS handling will cover the negative encoding.
    ops["nines_comp_sub"] = lambda a, b: z3.If(
        a >= b,
        a + 99 - b,       # positive branch: a ≥ b → result magnitude a+99-b
        -(a + 99 - b),    # negative branch: a < b → negative of the same magnitude
    )
    return ops


OPS = _op_builders()


# ── Prompt parsing ────────────────────────────────────────────────────────

def parse_row(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    out: list[tuple[str, str]] = []
    q: str | None = None
    for line in prompt.splitlines():
        ln = line.strip()
        m = re.search(r"determine the result for:\s*(\S+)", ln)
        if m:
            q = m.group(1)
            continue
        m = re.match(r"^(\S+)\s*=\s*(\S+)\s*$", ln)
        if m and not re.search(r"[A-Za-z]", m.group(1) + m.group(2)):
            out.append((m.group(1), m.group(2)))
    return out, q


# ── Template rules (S1, S2) ───────────────────────────────────────────────

TemplateElem = tuple[str, Any]  # ('pos', int) or ('lit', str)


def _enumerate_templates_for_length(
    op_examples: list[tuple[str, str]], rhs_len: int
) -> list[tuple[TemplateElem, ...]]:
    """Return all templates of length ``rhs_len`` explaining every example's RHS.

    A template is a tuple of ``rhs_len`` elements; each element is either
    ``('pos', k)`` (sources from LHS[k]) or ``('lit', c)`` (literal char c).
    """
    per_pos_sources: list[list[TemplateElem]] = []
    for j in range(rhs_len):
        rhs_chars = [rhs[j] for _, rhs in op_examples]
        sources: list[TemplateElem] = []
        # Position sources: any k where LHS[k] matches rhs[j] across all examples
        for k in range(LHS_LEN):
            if all(lhs[k] == rhs_chars[i] for i, (lhs, _) in enumerate(op_examples)):
                sources.append(("pos", k))
        # Literal source: only if rhs[j] is the same char across all examples
        if len(set(rhs_chars)) == 1:
            sources.append(("lit", rhs_chars[0]))
        if not sources:
            return []  # Position j unexplainable
        per_pos_sources.append(sources)
    # Cartesian product — cap at 6^5 = 7,776
    templates: list[tuple[TemplateElem, ...]] = []
    for combo in product(*per_pos_sources):
        templates.append(combo)
    return templates


def _apply_template(tpl: tuple[TemplateElem, ...], lhs: str) -> str:
    out = []
    for kind, val in tpl:
        if kind == "pos":
            out.append(lhs[val])
        else:
            out.append(val)
    return "".join(out)


def find_template_rules(
    op_examples: list[tuple[str, str]]
) -> list[tuple[TemplateElem, ...]]:
    """Per-op template rules covering all examples. Empty list = no template fits."""
    if not op_examples:
        return []
    rhs_lens = {len(rhs) for _, rhs in op_examples}
    if len(rhs_lens) != 1:
        return []  # Variable-length RHS → not a pure template
    (L,) = rhs_lens
    if L == 0:
        return [tuple()]  # empty RHS template trivially fits
    if L > 5:
        return []  # tractability cap
    return _enumerate_templates_for_length(op_examples, L)


def _tpl_desc(tpl: tuple[TemplateElem, ...]) -> str:
    parts = []
    for kind, val in tpl:
        if kind == "pos":
            parts.append(str(val))
        else:
            parts.append(f"'{val}'")
    return "[" + " ".join(parts) + "]"


# ── Arithmetic rules (S3) — op_family candidates per op ───────────────────

def _syms_to_z3(s: str, dvars: dict[str, z3.ExprRef]) -> z3.ExprRef:
    val = z3.IntVal(0)
    for c in s:
        val = val * 10 + dvars[c]
    return val


# Multiplication-family ops need a longer z3 budget because nonlinear
# arithmetic is hard — at short budgets z3 returns ``unknown`` rather than
# ``sat``. We treat those as candidates (the joint solver re-verifies).
_MUL_HARD_OPS = {
    "mul", "mul+1", "mul-1",
    "cross_mul", "cross_mul_rev",
    "determinant", "abs_determinant",
    "digit_mul", "digit_mul_rev",
    "digit_product_diff", "digit_product_sum",
    "nines_comp_sub",
}


def feasible_arith_families(
    op_examples: list[tuple[str, str]],
    time_budget_sec: float = 0.3,
) -> list[str]:
    """Per-op candidate arithmetic families.

    For each family, run a standalone z3 with only this op's examples to check
    if any bijection makes the examples consistent. The returned families are
    CANDIDATES; the final row-level z3 re-solves with all ops sharing a bijection.

    Nonlinear ops (multiplication family) can time out (``unknown``) at short
    budgets. We conservatively treat ``unknown`` as a candidate so the joint
    solver gets a chance to verify or rule them out.
    """
    if not op_examples:
        return []
    # Operand/RHS chars
    chars: set[str] = set()
    for lhs, rhs in op_examples:
        chars |= set(lhs[:OP_POS] + lhs[OP_POS + 1:] + rhs)
    if len(chars) > 10:
        return []  # No bijection over 0..9 possible here
    feasible: list[str] = []
    for fam, body in OPS.items():
        # Give nonlinear families extra budget so they return sat rather than unknown
        per_budget_sec = time_budget_sec * (4.0 if fam in _MUL_HARD_OPS else 1.0)
        s = z3.Solver()
        s.set("timeout", max(20, int(per_budget_sec * 1000)))
        dvars = {c: z3.Int(f"d_{i}") for i, c in enumerate(sorted(chars))}
        for v in dvars.values():
            s.add(v >= 0, v <= 9)
        s.add(z3.Distinct(*dvars.values()))
        for lhs, rhs in op_examples:
            left, right = lhs[:OP_POS], lhs[OP_POS + 1:]
            if len(left) > 1:
                s.add(dvars[left[0]] != 0)
            if len(right) > 1:
                s.add(dvars[right[0]] != 0)
            a, b = _syms_to_z3(left, dvars), _syms_to_z3(right, dvars)
            computed = body(a, b)
            positive = _syms_to_z3(rhs, dvars) == computed
            if len(rhs) >= 2:
                negative = z3.And(computed < 0, _syms_to_z3(rhs[1:], dvars) == -computed)
                s.add(z3.Or(positive, negative))
            else:
                s.add(positive)
        r = s.check()
        # Treat unknown (timeout) as candidate — joint solver will re-verify.
        if r == z3.sat or r == z3.unknown:
            feasible.append(fam)
    return feasible


# ── Row-level joint solver ────────────────────────────────────────────────

def _apply_reverse(examples: list[tuple[str, str]], q_lhs: str, q_rhs: str):
    """S4 preprocessor: reverse both LHS and RHS of every row."""
    rev_ex = [(lhs[::-1], rhs[::-1]) for lhs, rhs in examples]
    return rev_ex, q_lhs[::-1], q_rhs[::-1]


def _apply_operand_reverse(examples: list[tuple[str, str]], q_lhs: str, q_rhs: str):
    """S6 preprocessor: reverse each 2-digit operand individually AND reverse the RHS.

    Handles the "little-endian" family where the Kaggle rule is
    ``op(reverse(a), reverse(b))`` with the result stored reversed.
    E.g. ``59-75 = 83``: ``|95-57| = 38``, reversed → ``83``.

    For LHS length 5 with operator at position 2, this produces:
        (lhs[1]+lhs[0]) + lhs[2] + (lhs[4]+lhs[3])

    RHS is reversed in full.
    """
    def _rev_operands_lhs(s: str) -> str:
        if len(s) != LHS_LEN:
            return s
        return s[1] + s[0] + s[2] + s[4] + s[3]
    rev_ex = [(_rev_operands_lhs(lhs), rhs[::-1]) for lhs, rhs in examples]
    return rev_ex, _rev_operands_lhs(q_lhs), q_rhs[::-1]


def _joint_arith_solve(
    by_op: dict[str, list[tuple[str, str]]],
    op_family_choice: dict[str, str],  # op_sym -> family name (for arith ops)
    op_template_choice: dict[str, tuple[TemplateElem, ...]],  # op_sym -> template (for template ops)
    q_lhs: str,
    stored_rhs: str,
    time_budget_sec: float,
) -> tuple[bool, dict | None]:
    """Joint z3: bijection shared across all arithmetic ops in the row, plus Q.

    Template ops carry no arithmetic constraint but their chars may still be
    constrained by the shared bijection IF they also appear at arith positions.
    Since templates don't need digit semantics, we simply skip them here.
    """
    arith_ops = [op for op in by_op if op in op_family_choice]
    if not arith_ops:
        # Pure template row — verified separately in main flow
        return False, None
    # Collect chars over arith op examples + Q (if Q is arith)
    chars: set[str] = set()
    for op in arith_ops:
        for lhs, rhs in by_op[op]:
            chars |= set(lhs[:OP_POS] + lhs[OP_POS + 1:] + rhs)
    q_op = q_lhs[OP_POS]
    if q_op in op_family_choice:
        chars |= set(q_lhs[:OP_POS] + q_lhs[OP_POS + 1:] + stored_rhs)
    if len(chars) > 10 or not chars:
        return False, None
    s = z3.Solver()
    s.set("timeout", max(50, int(time_budget_sec * 1000)))
    dvars = {c: z3.Int(f"d_{i}") for i, c in enumerate(sorted(chars))}
    for v in dvars.values():
        s.add(v >= 0, v <= 9)
    s.add(z3.Distinct(*dvars.values()))
    for op in arith_ops:
        body = OPS[op_family_choice[op]]
        for lhs, rhs in by_op[op]:
            left, right = lhs[:OP_POS], lhs[OP_POS + 1:]
            if len(left) > 1:
                s.add(dvars[left[0]] != 0)
            if len(right) > 1:
                s.add(dvars[right[0]] != 0)
            a, b = _syms_to_z3(left, dvars), _syms_to_z3(right, dvars)
            computed = body(a, b)
            pos = _syms_to_z3(rhs, dvars) == computed
            if len(rhs) >= 2:
                neg = z3.And(computed < 0, _syms_to_z3(rhs[1:], dvars) == -computed)
                s.add(z3.Or(pos, neg))
            else:
                s.add(pos)
    # Q constraint
    if q_op in op_family_choice:
        body = OPS[op_family_choice[q_op]]
        left, right = q_lhs[:OP_POS], q_lhs[OP_POS + 1:]
        if len(left) > 1:
            s.add(dvars[left[0]] != 0)
        if len(right) > 1:
            s.add(dvars[right[0]] != 0)
        a, b = _syms_to_z3(left, dvars), _syms_to_z3(right, dvars)
        computed = body(a, b)
        pos = _syms_to_z3(stored_rhs, dvars) == computed
        if len(stored_rhs) >= 2:
            neg = z3.And(computed < 0, _syms_to_z3(stored_rhs[1:], dvars) == -computed)
            s.add(z3.Or(pos, neg))
        else:
            s.add(pos)
    elif q_op in op_template_choice:
        tpl = op_template_choice[q_op]
        predicted = _apply_template(tpl, q_lhs)
        if predicted != stored_rhs:
            return False, None
    else:
        return False, None
    if s.check() != z3.sat:
        return False, None
    m = s.model()
    bijection = {c: m[v].as_long() for c, v in dvars.items()}
    return True, {"bijection": bijection}


def _solve_q_op_only_arith(
    q_op_examples: list[tuple[str, str]],
    q_lhs: str,
    stored_rhs: str,
    fam: str,
    time_budget_sec: float,
) -> dict | None:
    """Q-op-only arithmetic check: run z3 with ONLY the Q-op's examples plus
    the Q constraint. Non-Q ops are ignored. Returns bijection dict or None.

    This lets us verify rows where other ops in the same row use rules outside
    our vocabulary — if the Q-op rule alone predicts stored_rhs, the label is
    consistent under that Q-op rule.
    """
    body = OPS[fam]
    chars: set[str] = set()
    for lhs, rhs in q_op_examples:
        chars |= set(lhs[:OP_POS] + lhs[OP_POS + 1:] + rhs)
    chars |= set(q_lhs[:OP_POS] + q_lhs[OP_POS + 1:] + stored_rhs)
    if len(chars) > 10 or not chars:
        return None
    s = z3.Solver()
    s.set("timeout", max(50, int(time_budget_sec * 1000)))
    dvars = {c: z3.Int(f"d_{i}") for i, c in enumerate(sorted(chars))}
    for v in dvars.values():
        s.add(v >= 0, v <= 9)
    s.add(z3.Distinct(*dvars.values()))
    for lhs, rhs in q_op_examples:
        left, right = lhs[:OP_POS], lhs[OP_POS + 1:]
        if len(left) > 1: s.add(dvars[left[0]] != 0)
        if len(right) > 1: s.add(dvars[right[0]] != 0)
        a, b = _syms_to_z3(left, dvars), _syms_to_z3(right, dvars)
        computed = body(a, b)
        pos = _syms_to_z3(rhs, dvars) == computed
        if len(rhs) >= 2:
            neg = z3.And(computed < 0, _syms_to_z3(rhs[1:], dvars) == -computed)
            s.add(z3.Or(pos, neg))
        else:
            s.add(pos)
    # Q constraint
    left, right = q_lhs[:OP_POS], q_lhs[OP_POS + 1:]
    if len(left) > 1: s.add(dvars[left[0]] != 0)
    if len(right) > 1: s.add(dvars[right[0]] != 0)
    a, b = _syms_to_z3(left, dvars), _syms_to_z3(right, dvars)
    computed = body(a, b)
    pos = _syms_to_z3(stored_rhs, dvars) == computed
    if len(stored_rhs) >= 2:
        neg = z3.And(computed < 0, _syms_to_z3(stored_rhs[1:], dvars) == -computed)
        s.add(z3.Or(pos, neg))
    else:
        s.add(pos)
    if s.check() != z3.sat:
        return None
    m = s.model()
    return {c: m[v].as_long() for c, v in dvars.items()}


def _solve_single_stage(
    examples: list[tuple[str, str]],
    q_lhs: str,
    stored_rhs: str,
    stage_tag: str,
    per_op_arith_budget: float,
    combo_budget: float,
) -> tuple[bool, dict | None]:
    """Stages S1/S2/S3/S5 (S4 = caller reverses row then calls us again)."""
    # Basic format checks
    if not examples or not q_lhs or len(q_lhs) != LHS_LEN:
        return False, None
    if any(len(lhs) != LHS_LEN for lhs, _ in examples):
        return False, None
    q_op = q_lhs[OP_POS]
    by_op: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for lhs, rhs in examples:
        by_op[lhs[OP_POS]].append((lhs, rhs))
    # Question operator must have appeared in examples (else we can't infer its rule)
    if q_op not in by_op:
        return False, None
    # Per-op candidate discovery
    op_templates: dict[str, list[tuple[TemplateElem, ...]]] = {}
    op_arith_families: dict[str, list[str]] = {}
    for op in by_op:
        op_templates[op] = find_template_rules(by_op[op])
        op_arith_families[op] = feasible_arith_families(by_op[op], per_op_arith_budget)
        if not op_templates[op] and not op_arith_families[op]:
            return False, None  # This op has no candidate — row unexplainable
    # Disambiguation guard: a template on an op with only 1 supporting example
    # is vacuous — literal slots can match any single RHS — so we only trust a
    # template on the Q-op if Q-op has ≥2 examples. Otherwise the Q-op must be
    # arith (or have its template predict stored via purely positional slots).
    q_op_examples = by_op[q_op]
    q_op_pos_only_templates = [
        t for t in op_templates[q_op] if all(k == "pos" for k, _ in t)
    ]
    q_op_can_template = (
        len(q_op_examples) >= 2 or bool(q_op_pos_only_templates)
    )
    # Which templates to try for Q-op: all if ≥2 examples, else only position-only
    q_op_tpl_candidates = (
        op_templates[q_op] if len(q_op_examples) >= 2 else q_op_pos_only_templates
    )
    # Non-Q ops contribute NO per-op template constraint to the row-level check
    # (they only affect the shared bijection when arith). For enumeration, pick
    # a single representative template per non-Q op (only the flag "template vs arith"
    # matters; the exact template is recorded for the witness only).
    op_list = [op for op in by_op if op != q_op] + [q_op]

    # Enumerate:
    #   Non-Q ops: choose kind in {template, arith_family}. Template flag carries no constraint.
    #   Q-op:     choose kind in {template (enumerated), arith_family}.
    def _kinds_for(op):
        if op == q_op:
            out = []
            for t in q_op_tpl_candidates:
                out.append(("t", t))
            for f in op_arith_families[op]:
                out.append(("a", f))
            return out
        out = []
        if op_templates[op]:
            out.append(("t", op_templates[op][0]))  # representative; value ignored by solver
        for f in op_arith_families[op]:
            out.append(("a", f))
        return out

    kinds_per_op = [_kinds_for(op) for op in op_list]
    if any(not k for k in kinds_per_op):
        return False, None

    # ── S1/S2: pure-template (no arith) shortcut for Q-op ──
    if q_op_can_template:
        for tpl_q in q_op_tpl_candidates:
            if _apply_template(tpl_q, q_lhs) == stored_rhs:
                # If all non-Q ops also have templates, we have a pure-template
                # witness with no bijection needed.
                if all(op_templates[op] for op in by_op if op != q_op):
                    non_q_tpls = {
                        op: op_templates[op][0] for op in by_op if op != q_op
                    }
                    non_q_tpls[q_op] = tpl_q
                    return True, {
                        "stage": stage_tag + ":template-only",
                        "templates": {op: _tpl_desc(t) for op, t in non_q_tpls.items()},
                    }
                # Q-op-only template witness (sibling ops not required to be explained).
                return True, {
                    "stage": stage_tag + ":q-op-only-template",
                    "templates": {q_op: _tpl_desc(tpl_q)},
                }

    # ── Q-op-only arithmetic fallback (before combinatorial mixed) ──
    # Runs one small z3 per candidate family — much cheaper than enumerating
    # template × arith choices over all ops.
    for fam in op_arith_families[q_op]:
        bij = _solve_q_op_only_arith(
            by_op[q_op], q_lhs, stored_rhs, fam,
            time_budget_sec=0.6,
        )
        if bij is not None:
            return True, {
                "stage": stage_tag + ":q-op-only-arith",
                "arith": {q_op: fam},
                "bijection": bij,
            }

    # ── S3/S5: arith (possibly mixed with templates for non-Q ops) ──
    # Hard cap on combo enumeration — the nonlinear joint z3 is the expensive
    # step; skipping this stage when it would explode beats hanging.
    combo_size = 1
    for k in kinds_per_op:
        combo_size *= len(k)
    if combo_size > 400:
        return False, None
    t0 = time.time()
    for choice_tuple in product(*kinds_per_op):
        if time.time() - t0 > combo_budget:
            return False, None
        op_family_choice: dict[str, str] = {}
        op_template_choice: dict[str, tuple[TemplateElem, ...]] = {}
        for op, (kind, val) in zip(op_list, choice_tuple):
            if kind == "t":
                op_template_choice[op] = val
            else:
                op_family_choice[op] = val
        # Skip pure-template combo (already tried above)
        if not op_family_choice:
            continue
        # Q's op template must predict stored
        if q_op in op_template_choice:
            if _apply_template(op_template_choice[q_op], q_lhs) != stored_rhs:
                continue
        ok, bij = _joint_arith_solve(
            by_op, op_family_choice, op_template_choice, q_lhs, stored_rhs,
            time_budget_sec=0.5,
        )
        if not ok:
            continue
        return True, {
            "stage": stage_tag + ":mixed",
            "templates": {op: _tpl_desc(t) for op, t in op_template_choice.items()},
            "arith": op_family_choice,
            "bijection": (bij or {}).get("bijection"),
        }

    return False, None


def solve_row(
    examples: list[tuple[str, str]],
    q_lhs: str,
    stored_rhs: str,
    time_budget_sec: float = 3.0,
) -> tuple[bool, dict | None]:
    """Full S1–S5 pipeline. Returns (consistent, witness_dict)."""
    # S1–S3 + S5
    per_op_budget = 0.25
    combo_budget = time_budget_sec * 0.6
    ok, w = _solve_single_stage(
        examples, q_lhs, stored_rhs,
        stage_tag="forward",
        per_op_arith_budget=per_op_budget,
        combo_budget=combo_budget,
    )
    if ok:
        return True, w
    # S4: mirror preprocessor — reverse whole row, then re-run S1/S2/S3/S5
    rev_ex, rev_q, rev_stored = _apply_reverse(examples, q_lhs, stored_rhs)
    ok, w = _solve_single_stage(
        rev_ex, rev_q, rev_stored,
        stage_tag="mirror",
        per_op_arith_budget=per_op_budget,
        combo_budget=time_budget_sec * 0.3,
    )
    if ok:
        return True, w
    # S6: operand-reverse preprocessor — reverse each 2-digit operand AND RHS,
    # covering the "little-endian" family (op(reverse(a),reverse(b)) with
    # reversed result), then re-run S1/S2/S3/S5.
    opr_ex, opr_q, opr_stored = _apply_operand_reverse(examples, q_lhs, stored_rhs)
    ok, w = _solve_single_stage(
        opr_ex, opr_q, opr_stored,
        stage_tag="operand_reverse",
        per_op_arith_budget=per_op_budget,
        combo_budget=time_budget_sec * 0.3,
    )
    if ok:
        return True, w
    return False, None


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="balanced_dev600.csv")
    ap.add_argument("--output", required=True, type=Path, help="symbolic_consistency.jsonl")
    ap.add_argument("--domain", default="equations")
    ap.add_argument("--time-budget", type=float, default=3.0, help="per-row seconds")
    ap.add_argument("--only-unverified", type=Path, default=None,
                    help="optional equations_consistency.jsonl — restrict to rows flagged inconsistent there")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.input, newline="")) if r["domain"] == args.domain]
    if args.only_unverified:
        prev = {json.loads(l)["id"]: json.loads(l) for l in open(args.only_unverified)}
        rows = [r for r in rows if r["id"] in prev and not prev[r["id"]]["consistent"]]
    print(f"checking {len(rows)} {args.domain} rows (budget {args.time_budget}s/row)")

    verdicts: list[dict[str, Any]] = []
    n_ok = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        examples, q_lhs = parse_row(r["prompt"])
        consistent, witness = False, None
        if examples and q_lhs:
            consistent, witness = solve_row(examples, q_lhs, r["answer"], args.time_budget)
        verdicts.append({
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
        for v in verdicts:
            f.write(json.dumps(v) + "\n")
    print(f"\nconsistent:   {n_ok}/{len(rows)}  ({100*n_ok/len(rows):.1f}%)")
    print(f"inconsistent: {len(rows)-n_ok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
