"""Consistency-based verifier for bit_manipulation labels.

The bit-rule puzzle is under-determined: with N examples and 8 output bits,
many per-bit rule families (identity/NOT, pair AND/OR/XOR, majority over 3,
MUX/choice) give the same example outputs but differ on the question input.
A deterministic "pick one rule" solver can fit the examples and still
disagree with the stored answer because it picked a different rule than
Kaggle's generator did — without the label being wrong.

This verifier asks a stricter question per output bit:

    Does there exist AT LEAST ONE rule in a broad Boolean family F such that
    the rule matches every example's output bit AND matches the stored answer
    on the question's output bit?

If yes for all 8 output bits → the stored label is consistent with the
examples under some rule in F → the label is trustworthy (strictly, we
can't say it's "the" correct rule, but we can say it's coherent).

If no for any bit → the stored label is NOT consistent with any rule in F
given the examples → the label is provably inconsistent (under F).

F covers (per-bit rules, enumerated over all index choices):
  * constants 0, 1
  * I_k for k in 0..7  (identity)
  * NOT I_k
  * AND/OR/XOR(I_a, I_b)        for all pairs, with optional NOT on each input
  * MAJ(I_a, I_b, I_c)          (majority over 3)
  * MUX(I_s, I_a, I_b)          (choice / if-then-else)
  * NOT wrapper on any of the above

Plus GLOBAL (whole-byte) transforms applied to all 8 output bits uniformly:
  * rotate left/right by 1..7
  * shift left/right by 1..7 with zero-fill
  * bit reverse
  * NOT wrapping any of the above
  * XOR with a constant mask learned from the examples

This widens the "consistent under F" set to cover Kaggle's multi-step
rules without requiring a multi-step solver.

Any bits label that's inconsistent with this family is extremely suspicious.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Callable


# Signature: Callable[[tuple[int,...]], int]  — takes 8 input bits, returns 0 or 1
Rule = Callable[[tuple], int]


def _bit(val: str | int, i: int) -> int:
    """Return the i-th bit (left-to-right by convention used in default traces)."""
    if isinstance(val, int):
        s = f"{val:08b}"
    else:
        s = val
    return int(s[i])


def _bits_of(s: str) -> tuple:
    return tuple(int(c) for c in s)


# Build the full rule family as a list of (name, rule_fn).
# ~900 rules total; cheap to brute-force.
def _rule_family() -> list[tuple[str, Rule]]:
    fam: list[tuple[str, Rule]] = []
    fam.append(("0", lambda b: 0))
    fam.append(("1", lambda b: 1))
    for k in range(8):
        fam.append((f"I{k}", (lambda b, k=k: b[k])))
        fam.append((f"!I{k}", (lambda b, k=k: 1 - b[k])))
    for a, c in combinations(range(8), 2):
        for neg_a, neg_c in product([False, True], repeat=2):
            def mk(a=a, c=c, na=neg_a, nc=neg_c):
                def _and(b):
                    va = b[a] if not na else 1 - b[a]
                    vc = b[c] if not nc else 1 - b[c]
                    return va & vc
                def _or(b):
                    va = b[a] if not na else 1 - b[a]
                    vc = b[c] if not nc else 1 - b[c]
                    return va | vc
                def _xor(b):
                    va = b[a] if not na else 1 - b[a]
                    vc = b[c] if not nc else 1 - b[c]
                    return va ^ vc
                return _and, _or, _xor
            fa, fo, fx = mk()
            prefix = ("!" if neg_a else "") + f"I{a}"
            suffix = ("!" if neg_c else "") + f"I{c}"
            fam.append((f"AND({prefix},{suffix})", fa))
            fam.append((f"OR({prefix},{suffix})", fo))
            fam.append((f"XOR({prefix},{suffix})", fx))
    # k-input AND/OR/XOR/MAJ with arbitrary NOT pattern on each input, k=3..5.
    # This covers Kaggle's observed rules like "maj(I0, !I1, I2, !I4, I6)".
    for k in (3, 4, 5):
        for idx in combinations(range(8), k):
            for neg_mask in range(1 << k):
                negs = tuple((neg_mask >> j) & 1 for j in range(k))
                def make(idx=idx, negs=negs, k=k):
                    def _val(b):
                        return tuple(b[idx[j]] ^ negs[j] for j in range(k))
                    def _and(b):
                        v = _val(b)
                        out = 1
                        for x in v:
                            out &= x
                        return out
                    def _or(b):
                        v = _val(b)
                        out = 0
                        for x in v:
                            out |= x
                        return out
                    def _xor(b):
                        v = _val(b)
                        out = 0
                        for x in v:
                            out ^= x
                        return out
                    def _maj(b):
                        v = _val(b)
                        s = sum(v)
                        return 1 if 2 * s > k else 0
                    return _and, _or, _xor, _maj
                fa, fo, fx, fm = make()
                arglist = ",".join(f"{'!' if negs[j] else ''}I{idx[j]}" for j in range(k))
                fam.append((f"AND({arglist})", fa))
                fam.append((f"OR({arglist})", fo))
                fam.append((f"XOR({arglist})", fx))
                fam.append((f"MAJ({arglist})", fm))
    # MUX: b[s] ? b[a] : b[b_] with optional NOT on selector
    for s, a, b_ in combinations(range(8), 3):
        for neg_s in (False, True):
            def mux(b, s=s, a=a, b_=b_, ns=neg_s):
                sel = b[s] ^ (1 if ns else 0)
                return b[a] if sel else b[b_]
            name = f"MUX({'!' if neg_s else ''}I{s},I{a},I{b_})"
            fam.append((name, mux))
    # NOT-wrapper on all previously-built rules (doubles the family but cheap
    # and catches "negate the selected function" puzzles).
    orig = list(fam)
    for name, r in orig:
        if name in ("0", "1") or name.startswith("I") or name.startswith("!I"):
            continue
        fam.append((f"!{name}", (lambda b, r=r: 1 - r(b))))
    return fam


RULE_FAMILY: list[tuple[str, Rule]] = _rule_family()


def _global_byte_transforms() -> list[tuple[str, Callable[[tuple], tuple]]]:
    """Whole-byte rules applied to a full 8-bit input; return the full output.

    Each transform: (name, byte_in: tuple[int,...]) -> byte_out: tuple[int,...]
    """
    out: list[tuple[str, Callable[[tuple], tuple]]] = []
    out.append(("identity", lambda b: b))
    out.append(("not", lambda b: tuple(1 - x for x in b)))
    out.append(("reverse", lambda b: b[::-1]))
    out.append(("not+reverse", lambda b: tuple(1 - x for x in b[::-1])))
    # rotates
    for n in range(1, 8):
        out.append((f"rol{n}", (lambda b, n=n: b[n:] + b[:n])))
        out.append((f"ror{n}", (lambda b, n=n: b[-n:] + b[:-n])))
        out.append((f"not+rol{n}", (lambda b, n=n: tuple(1 - x for x in (b[n:] + b[:n])))))
        out.append((f"not+ror{n}", (lambda b, n=n: tuple(1 - x for x in (b[-n:] + b[:-n])))))
    # logical shifts (zero-fill)
    for n in range(1, 8):
        out.append((f"shl{n}", (lambda b, n=n: b[n:] + (0,) * n)))
        out.append((f"shr{n}", (lambda b, n=n: (0,) * n + b[:-n])))
    # XOR with every 8-bit mask  (256 rules) — catches "flip these specific bits"
    for mask in range(256):
        out.append((f"xor{mask:08b}", (lambda b, mask=mask:
                    tuple(x ^ ((mask >> (7 - i)) & 1) for i, x in enumerate(b)))))
    return out


GLOBAL_TRANSFORMS: list[tuple[str, Callable[[tuple], tuple]]] = _global_byte_transforms()


def _try_global_transform(
    ex_inputs: list[tuple],
    ex_outputs: list[tuple],
    q_in: tuple,
    stored_out: tuple,
) -> str | None:
    """Return a global rule name that matches every example AND the stored Q answer, or None."""
    for name, fn in GLOBAL_TRANSFORMS:
        if all(fn(i) == o for i, o in zip(ex_inputs, ex_outputs)) and fn(q_in) == stored_out:
            return name
    return None


def _try_truth_table_rule(
    ex_inputs: list[tuple],
    target_examples: list[int],
    q_in: tuple,
    target_q: int,
    max_k: int = 4,
) -> str | None:
    """Find any index-subset (size k in 1..max_k) whose 2^k truth-table entries
    are simultaneously consistent with every example and the stored Q bit.

    This is broader than named Boolean functions — it covers every possible
    k-input function. 3-input puzzles that weren't catchable by MAJ/MUX/etc.
    get caught here.
    """
    from itertools import combinations
    for k in range(1, max_k + 1):
        for idx in combinations(range(8), k):
            constraints: dict[tuple, int] = {}
            ok = True
            for ex_in, t in zip(ex_inputs, target_examples):
                v = tuple(ex_in[j] for j in idx)
                if v in constraints and constraints[v] != t:
                    ok = False
                    break
                constraints[v] = t
            if not ok:
                continue
            qv = tuple(q_in[j] for j in idx)
            if qv in constraints and constraints[qv] != target_q:
                continue
            # Found a consistent subset + truth-table
            return f"TT({','.join(f'I{i}' for i in idx)})"
    return None


def is_bits_label_consistent(
    examples: list[tuple[str, str]],
    question_input: str,
    stored_output: str,
) -> tuple[bool, list[str]]:
    """Check whether stored_output is consistent with examples under F.

    Two stage check:
      1. Is there a single GLOBAL byte transform (rotate/shift/xor-mask/reverse)
         that maps every example input to its example output AND maps the
         question input to the stored answer? If so, label is consistent via
         a whole-byte rule (the witness list will all be the same global name).
      2. Otherwise, is there a per-bit rule from the local family (constants,
         identity/NOT, 2-arg boolean, MAJ, MUX) for EACH output bit separately
         that matches every example's output bit AND the stored bit on q_in?
         If all 8 bits have per-bit witnesses, label is consistent.

    Returns (consistent, per_bit_witnesses).
    """
    ex_inputs = [_bits_of(i) for i, _ in examples]
    ex_outputs = [_bits_of(o) for _, o in examples]
    q_in = _bits_of(question_input)
    stored = _bits_of(stored_output)

    # Stage 1: global byte transform
    g = _try_global_transform(ex_inputs, ex_outputs, q_in, stored)
    if g is not None:
        return True, [g] * 8

    # Stage 2: per-bit named rule (fast path)
    witnesses: list[str] = []
    ok = True
    for bit in range(8):
        target_examples = [ex_o[bit] for ex_o in ex_outputs]
        target_q = stored[bit]
        found: str | None = None
        shortest = None
        for name, rule in RULE_FAMILY:
            if not all(rule(ex_in) == t for ex_in, t in zip(ex_inputs, target_examples)):
                continue
            if rule(q_in) != target_q:
                continue
            if shortest is None or len(name) < len(shortest):
                shortest = name
                found = name
        if found is not None:
            witnesses.append(found)
            continue
        # Stage 3: generic k-input truth table (catches the long tail)
        tt = _try_truth_table_rule(ex_inputs, target_examples, q_in, target_q, max_k=4)
        if tt is not None:
            witnesses.append(tt)
            continue
        ok = False
        witnesses.append("<no-consistent-rule>")
    return ok, witnesses


# ── Uniform per-domain API ────────────────────────────────────────────────
# Every Kaggle-domain verifier under this package exposes the same shape:
#
#     parse(prompt, stored_answer="", _id="") -> Problem | None
#     verify(prompt, stored_answer)           -> dict
#
# For ``bits``, ``verify`` runs the stricter consistency check
# (``is_bits_label_consistent``) — the puzzle is under-determined, so
# "label agrees with examples under SOME rule in the family" is the right
# semantic rather than "label matches the unique solver guess".

DOMAIN = "bits"


def _parse_bits_prompt(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    import re as _re
    pairs = _re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    q = _re.search(r"determine the output for:\s*([01]{8})", prompt)
    return pairs, (q.group(1) if q else None)


def parse(prompt: str, stored_answer: str = "", _id: str = ""):
    from agent_evolve.model.data.reasoners.store_types import Example, Problem
    pairs, q = _parse_bits_prompt(prompt)
    if not pairs or not q:
        return None
    return Problem(
        id=_id,
        category="bit_manipulation",
        examples=[Example(i, o) for i, o in pairs],
        question=q,
        answer=stored_answer,
        prompt=prompt,
    )


def verify(prompt: str, stored_answer: str) -> dict:
    """Stricter consistency-based verifier for bits labels.

    Returns {"domain", "agrees", "prediction", "status", "witness"}. When the
    label is consistent under some rule in family F, ``agrees`` is True and
    ``witness`` lists the per-bit rule(s) that explain it.
    """
    base = {"domain": DOMAIN, "agrees": False, "prediction": None,
            "status": "", "witness": None}
    pairs, q = _parse_bits_prompt(prompt)
    if not pairs or not q:
        return {**base, "status": "parse_failed"}
    try:
        consistent, witnesses = is_bits_label_consistent(pairs, q, stored_answer)
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": f"verifier_error: {exc!r}"}
    return {
        "domain": DOMAIN,
        "agrees": bool(consistent),
        "prediction": stored_answer if consistent else None,
        "status": "ok",
        "witness": witnesses if consistent else None,
    }


# ── CLI entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import csv, json, re, sys, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--domain", default="bits")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input, newline="")))
    rows = [r for r in rows if r["domain"] == args.domain]
    print(f"checking {len(rows)} bits rows")

    results = []
    n_consistent = 0
    n_inconsistent = 0
    for r in rows:
        pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", r["prompt"])
        q = re.search(r"determine the output for:\s*([01]{8})", r["prompt"]).group(1)
        consistent, witnesses = is_bits_label_consistent(pairs, q, r["answer"])
        results.append({
            "id": r["id"], "source": r.get("source", ""),
            "stored_answer": r["answer"],
            "consistent": consistent,
            "witness_rules": witnesses,
        })
        if consistent: n_consistent += 1
        else: n_inconsistent += 1

    with open(args.output, "w") as f:
        for v in results:
            f.write(json.dumps(v) + "\n")

    print(f"\nconsistent:   {n_consistent}  ({100*n_consistent/len(rows):.1f}%)")
    print(f"inconsistent: {n_inconsistent}  ({100*n_inconsistent/len(rows):.1f}%)")
    if n_inconsistent:
        print("\nfirst 5 inconsistent rows (label possibly wrong):")
        bad = [v for v in results if not v["consistent"]][:5]
        for v in bad:
            print(f"  id={v['id']} stored={v['stored_answer']} "
                  f"bad_bits={[i for i,w in enumerate(v['witness_rules']) if w.startswith('<')]}")
