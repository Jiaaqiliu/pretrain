"""Per-category data-efficiency leaderboard built from ablation_report records.

Reads `ablation_report` records from the active workspace's ledger,
parses each body for arm_a / arm_b accuracies + delta + verdict, and
prints a table grouped by category. Categories with no ablation yet
are listed explicitly so the user can see what's missing.

Usage:
    python -m agent_evolve.model.data.pipelines.legacy.shared.leaderboard \
        [--workspace <root>] [--category <cat>]

Read-only — does NOT write to memory. Useful for the planner and for
human inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .stages.witness_search import DOMAIN_HEURISTIC


# Pull "<cat>.acc: 0.????" lines for both arms out of the body.
# The ablation_report body shape is documented in
# `.claude/skills/trainer-ablation-collect/SKILL.md` step 3.
_ARM_ACC_RE = re.compile(
    r"(?ms)^arm_(a|b)\s*\([^)]*\):\s*\n"          # arm header
    r"(?:[^\n]*\n)*?"                              # any leading lines (training_run_id, data, …)
    r"\s*(\w+)\.acc:\s*([0-9.]+)\b"                # <cat>.acc: 0.123
)

_DELTA_RE = re.compile(r"(?m)^delta_\w+_acc\s*=\s*[^=]*=\s*([+-]?[0-9.]+)\s*$")
_VERDICT_RE = re.compile(r"(?m)^verdict:\s*(\w+)")
_CATEGORY_RE = re.compile(r"(?m)^category:\s*(\w+)")


def _run(cmd: list[str]) -> dict:
    """Run a CLI subcommand; return the LAST JSON line on stdout."""
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\n{out.stderr.strip()}")
    last_line = out.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def _fetch_records(top_k: int) -> list[dict]:
    """Fetch up to `top_k` ablation_report records via the nemo_mas CLI.

    Uses `mem recent --kind ablation_report` because we want the
    leaderboard sorted by recency. The CLI returns full bodies inline.
    """
    cmd = [
        sys.executable, "-m", "agent_evolve.model.algorithms.nemo_mas.cli",
        "mem", "recent", "--kind", "ablation_report", "-k", str(top_k),
    ]
    obj = _run(cmd)
    return obj.get("records", []) if obj.get("ok") else []


def _parse_body(body: str) -> dict | None:
    cat_m = _CATEGORY_RE.search(body)
    if not cat_m:
        return None
    cat = cat_m.group(1)

    arm_a_acc = arm_b_acc = None
    for m in _ARM_ACC_RE.finditer(body):
        arm, line_cat, val = m.group(1), m.group(2), float(m.group(3))
        # Only the on-category line counts (the body also lists a "full
        # breakdown" with cross-domain accs that we ignore here).
        if line_cat != cat:
            continue
        if arm == "a" and arm_a_acc is None:
            arm_a_acc = val
        elif arm == "b" and arm_b_acc is None:
            arm_b_acc = val

    delta_m = _DELTA_RE.search(body)
    verdict_m = _VERDICT_RE.search(body)
    return {
        "category": cat,
        "arm_a_acc": arm_a_acc,
        "arm_b_acc": arm_b_acc,
        "delta": float(delta_m.group(1)) if delta_m else None,
        "verdict": verdict_m.group(1) if verdict_m else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=None,
                    help="override NEMO_MAS_WORKSPACE_ROOT")
    ap.add_argument("--category", default=None,
                    help="filter to a single category (default: all)")
    ap.add_argument("--top-k", type=int, default=50,
                    help="how many recent ablation_reports to scan")
    args = ap.parse_args(argv)

    if args.workspace:
        os.environ["NEMO_MAS_WORKSPACE_ROOT"] = args.workspace
        os.environ.setdefault("NEMO_MAS_WORK_DIR", args.workspace)
        os.environ.setdefault("NEMO_MAS_MEMORY_PATH",
                              str(Path(args.workspace) / "memory" / "records.jsonl"))

    records = _fetch_records(args.top_k)

    # Latest record per category wins.
    latest_by_cat: dict[str, dict] = {}
    for rec in records:
        parsed = _parse_body(rec.get("body", ""))
        if not parsed:
            continue
        cat = parsed["category"]
        if cat in latest_by_cat:
            continue   # records arrive newest-first
        latest_by_cat[cat] = {
            **parsed,
            "record_id": rec.get("id", "?"),
            "ts": rec.get("ts", "?"),
            "title": rec.get("title", ""),
        }

    cats = sorted(DOMAIN_HEURISTIC.keys())
    if args.category:
        cats = [args.category]

    # Header + rows
    fmt = "  {cat:<10}  {a:>9}  {b:>9}  {d:>8}  {v:<10}  {rid:<18}  {ts:<20}"
    print()
    print("Per-category data-efficiency leaderboard")
    print("(arm_a = baseline subset of default_14718, arm_b = curated_clean_<cat>)")
    print()
    print(fmt.format(
        cat="category", a="arm_a", b="arm_b", d="delta",
        v="verdict", rid="record_id", ts="ts",
    ))
    print("  " + "-" * 92)
    for cat in cats:
        row = latest_by_cat.get(cat)
        if row is None:
            print(fmt.format(
                cat=cat, a="—", b="—", d="—",
                v="(none yet)", rid="—", ts="—",
            ))
            continue
        a = f"{row['arm_a_acc']:.4f}" if row["arm_a_acc"] is not None else "—"
        b = f"{row['arm_b_acc']:.4f}" if row["arm_b_acc"] is not None else "—"
        d = (f"{row['delta']:+.4f}" if row["delta"] is not None else "—")
        v = row["verdict"] or "—"
        print(fmt.format(
            cat=cat, a=a, b=b, d=d, v=v,
            rid=row["record_id"], ts=row["ts"],
        ))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
