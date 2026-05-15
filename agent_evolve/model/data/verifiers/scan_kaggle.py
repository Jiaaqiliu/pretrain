"""Scan the full Kaggle ``train.csv`` (``id,prompt,answer``, no ``domain``
column) with every verifier we have, recording per-row verdicts +
per-domain coverage.

For each row:

  1. Infer domain from prompt text (see :func:`verifiers.infer_domain`).
  2. Dispatch to that domain's verifier (see :data:`verifiers.VERIFIERS`).

Output: a JSONL with one row per input + a summary printed to stdout.

Per-domain semantics:

  * ``cipher`` / ``gravity`` / ``numerals`` / ``units`` — run the default
    ``reasoning_<domain>`` solver and check its ``\\boxed{}`` equals the
    stored answer ("solver-agrees" → label is consistent).
  * ``bits`` — stricter consistency check (``is_bits_label_consistent``):
    label is consistent with examples under some rule in the broad family.
  * ``equations`` — cascading verifier (arithmetic z3 → S1–S5 symbolic):
    label is consistent under some rule in the equations family.

Usage::

    python -m agent_evolve.model.data.verifiers.scan_kaggle \\
        --input /tmp/kaggle_comp/train.csv \\
        --output runs/.../scan/kaggle_train.scan.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, "/fsx/zzsamshi/a-evolve")

from agent_evolve.model.data.verifiers import VERIFIERS, infer_domain  # noqa: E402


def scan_row(row: dict[str, str], eq_time_budget: float) -> dict[str, Any]:
    rid = row["id"]
    prompt = row["prompt"]
    answer = row["answer"]
    domain = infer_domain(prompt)
    base = {"id": rid, "domain": domain, "stored_answer": answer}
    fn = VERIFIERS.get(domain)
    if fn is None:
        if domain in VERIFIERS:
            return {**base, "method": "none", "status": "verifier_unavailable", "agrees": False}
        return {**base, "method": "none", "status": "unknown_domain", "agrees": False}
    if domain == "equations":
        verdict = fn(prompt, answer, time_budget_sec=eq_time_budget)  # type: ignore[call-arg]
    else:
        verdict = fn(prompt, answer)
    return {**base,
            "method": verdict.get("method") or verdict.get("domain", domain),
            "status": verdict.get("status", ""),
            "agrees": bool(verdict.get("agrees")),
            "prediction": verdict.get("prediction"),
            "witness": verdict.get("witness")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="/tmp/kaggle_comp/train.csv", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eq-time-budget", type=float, default=3.0,
                    help="per-row seconds for the equations verifier cascade")
    ap.add_argument("--domain", default=None, help="restrict to one Kaggle domain")
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
                      + "  ".join(f"{k}:{v['ok']}/{v['n']}"
                                  for k, v in sorted(per_domain.items())))

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
