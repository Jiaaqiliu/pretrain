"""Programmatic label verification: dataset-level scan that applies the
default per-domain solver as a ground-truth oracle and reports per-domain
agreement vs. the stored ``answer``.

Reads a CSV with columns ``id, prompt, answer, domain`` (the shape produced
by ``data/scripts/build_dev_set``). Writes a JSONL of per-row verdicts plus
a per-domain summary on stdout.

The per-domain logic lives in :mod:`agent_evolve.model.data.verifiers.<domain>`
— this file is just a runner over the
:data:`agent_evolve.model.data.verifiers.VERIFIERS` map.

Disagreement indicates either a label error in the dataset OR a solver
that couldn't handle a particular example edge case.

Usage::

    python -m agent_evolve.model.data.verifiers.programmatic \\
        --input runs/.../eval/balanced_dev600.csv \\
        --output runs/.../eval/balanced_dev600.programmatic.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/fsx/zzsamshi/a-evolve")

from agent_evolve.model.data.verifiers import VERIFIERS  # noqa: E402

logger = logging.getLogger(__name__)


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
            rows.append(Row(
                id=r["id"],
                prompt=r["prompt"],
                answer=r["answer"],
                domain=r.get("domain", ""),
                source=r.get("source", ""),
            ))
            if limit and len(rows) >= limit:
                break
    return rows


def judge_row(r: Row) -> dict:
    """Verify a single row by dispatching to the domain's verifier."""
    base = {
        "id": r.id, "domain": r.domain, "source": r.source,
        "stored_answer": r.answer,
    }
    fn = VERIFIERS.get(r.domain)
    if fn is None:
        return {**base, "solver_prediction": None,
                "solver_status": "no_solver" if r.domain not in VERIFIERS else "verifier_unavailable",
                "agrees": None}
    v = fn(r.prompt, r.answer)
    return {
        **base,
        "solver_prediction": v.get("prediction"),
        "solver_status": v.get("status", ""),
        "agrees": v.get("agrees"),
        "witness": v.get("witness"),
        "method": v.get("method"),
    }


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

    per: dict[str, dict[str, int]] = defaultdict(lambda: {
        "n": 0, "ok": 0, "agree": 0, "disagree": 0,
        "parse_fail": 0, "no_solution": 0, "no_boxed": 0, "errors": 0,
    })
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

    print("\n=== programmatic label verification (default solvers / verifiers) ===")
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
            print(f"  id={v['id']} dom={v['domain']:10s} src={v.get('source',''):13s} "
                  f"stored={v['stored_answer']!r}  solver={v['solver_prediction']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
