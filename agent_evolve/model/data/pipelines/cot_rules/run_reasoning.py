"""Generate deterministic reasoning text for each problem.

Walks ``problems.jsonl`` (Kaggle category names) and dispatches each
entry to its programmatic reasoner under
:mod:`agent_evolve.model.data.reasoners`. The CoT trace is written to
``reasoning/<id>.txt``; the in-place updated index records which problems
ended up with ``status: rule_found`` versus ``rule_unknown``.

This is a faithful port of huikang's ``reasoning.py`` adapted to our
package layout. Reasoners produce identical output text — only file paths
and module names changed.

Usage::

    cd <work_dir>      # must contain problems.jsonl + problems/<id>.jsonl
    python -m agent_evolve.model.data.pipelines.cot_rules.run_reasoning
    python -m agent_evolve.model.data.pipelines.cot_rules.run_reasoning --delete-investigations
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ...reasoners.bits import solve as reasoning_bit_manipulation
from ...reasoners.cipher import solve as reasoning_cipher
from ...reasoners.cryptarithm import reasoning_cryptarithm
from ...reasoners.equation_numeric import reasoning_equation_numeric
from ...reasoners.gravity import solve as reasoning_gravity
from ...reasoners.numerals import solve as reasoning_numeral
from ...reasoners.store_types import Problem
from ...reasoners.units import solve as reasoning_unit_conversion

PROBLEMS_INDEX = Path("problems.jsonl")
REASONING_DIR = Path("reasoning")
INVESTIGATIONS_DIR = Path("investigations")
INVESTIGATION_CATEGORIES: set[str] = {
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
}

SKIP_CATEGORIES: set[str] = set()

GENERATORS: dict[str, Callable] = {
    "gravity":                  reasoning_gravity,
    "unit_conversion":          reasoning_unit_conversion,
    "cipher":                   reasoning_cipher,
    "bit_manipulation":         reasoning_bit_manipulation,
    "numeral":                  reasoning_numeral,
    "equation_numeric_deduce":  reasoning_equation_numeric,
    "equation_numeric_guess":   reasoning_equation_numeric,
    "cryptarithm_deduce":       reasoning_cryptarithm,
    "cryptarithm_guess":        reasoning_cryptarithm,
}


def extract_answer(reasoning_text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", reasoning_text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def compare_answer(stored_answer: str, predicted: str) -> bool:
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


@dataclass
class CategoryCounts:
    rule_found: int = 0
    total: int = 0
    runtimes: list[float] = field(default_factory=list)


def run(
    work_dir: Path | str | None = None,
    *,
    delete_investigations: bool = False,
    categories: list[str] | None = None,
) -> None:
    """Programmatic entry point. Use ``main()`` for CLI.

    ``work_dir`` is cd'd into before reading ``problems.jsonl`` etc; if
    None, run in the current cwd. ``categories`` is a whitelist; ``None``
    means run every category in :data:`GENERATORS`.
    """
    import os
    prev_cwd = os.getcwd()
    if work_dir is not None:
        os.chdir(work_dir)
    try:
        return _run_in_cwd(delete_investigations=delete_investigations,
                           categories=categories)
    finally:
        os.chdir(prev_cwd)


def _run_in_cwd(*, delete_investigations: bool,
                categories: list[str] | None) -> None:
    if not PROBLEMS_INDEX.exists():
        print(f"No {PROBLEMS_INDEX} found.")
        return

    existing: dict[str, dict] = {}
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                existing[entry["id"]] = entry

    if REASONING_DIR.exists():
        shutil.rmtree(REASONING_DIR)
    REASONING_DIR.mkdir(parents=True)
    INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)

    stats: dict[str, bool] = {}
    category_stats: dict[str, CategoryCounts] = {}
    generated = 0
    skipped = 0

    for entry in existing.values():
        pid = entry["id"]
        category = entry["category"]

        if category not in category_stats:
            category_stats[category] = CategoryCounts()
        category_stats[category].total += 1

        if category in SKIP_CATEGORIES:
            existing[pid]["status"] = "rule_unknown"
            existing[pid]["submission"] = ""
            continue
        if categories is not None and category not in categories:
            existing[pid]["status"] = "rule_unknown"
            existing[pid]["submission"] = ""
            continue

        generator = GENERATORS.get(category)
        if not generator:
            existing[pid]["status"] = "rule_unknown"
            existing[pid]["submission"] = ""
            continue

        problem = Problem.load_from_json(pid)
        t0 = time.perf_counter()
        reasoning_text = generator(problem)
        elapsed = time.perf_counter() - t0
        category_stats[category].runtimes.append(elapsed)

        if reasoning_text is None:
            skipped += 1
            existing[pid]["status"] = "rule_unknown"
            existing[pid]["submission"] = ""
            continue

        submission = extract_answer(reasoning_text)
        result = compare_answer(problem.answer, submission)
        stats[pid] = result
        existing[pid]["status"] = "rule_found" if result else "rule_unknown"
        existing[pid]["submission"] = submission

        if result:
            category_stats[category].rule_found += 1

        out_path = REASONING_DIR / f"{pid}.txt"
        with open(out_path, "w") as f:
            f.write(reasoning_text)

        if category in INVESTIGATION_CATEGORIES:
            inv_path = INVESTIGATIONS_DIR / f"{pid}.txt"
            if result and delete_investigations and inv_path.exists():
                inv_path.unlink()

        generated += 1

    hypothesis_formed = 0
    for inv_path in INVESTIGATIONS_DIR.glob("*.txt"):
        pid = inv_path.stem
        if pid not in existing:
            continue
        if existing[pid]["status"] == "rule_found":
            continue
        existing[pid]["status"] = "hypothesis_formed"
        hypothesis_formed += 1

    with PROBLEMS_INDEX.open("w") as f:
        for entry in existing.values():
            entry.pop("has_investigation", None)
            f.write(json.dumps(entry) + "\n")

    total = sum(c.total for c in category_stats.values())
    rule_found = sum(c.rule_found for c in category_stats.values())
    print(f"\nGenerated {generated} reasoning files in {REASONING_DIR}/")
    if skipped:
        print(f"Skipped {skipped} (no generator for category)")
    if hypothesis_formed:
        print(f"Hypothesis formed: {hypothesis_formed} (investigation without reasoning)")
    w = 64
    print(f"\n{'=' * w}")
    print(f"{'Category':<28} {'Found':>6} {'Total':>6} {'Accuracy':>10} {'Avg ms':>10}")
    print(f"{'-' * w}")
    all_runtimes: list[float] = []
    for category_name, counts in sorted(category_stats.items()):
        acc = counts.rule_found / counts.total * 100 if counts.total else 0
        avg_ms = (
            sum(counts.runtimes) / len(counts.runtimes) * 1000 if counts.runtimes else 0
        )
        all_runtimes.extend(counts.runtimes)
        print(
            f"{category_name:<28} {counts.rule_found:>6} {counts.total:>6} {acc:>9.1f}% {avg_ms:>10.1f}"
        )
    print(f"{'-' * w}")
    overall_acc = rule_found / total * 100 if total else 0
    overall_avg_ms = sum(all_runtimes) / len(all_runtimes) * 1000 if all_runtimes else 0
    print(f"{'TOTAL':<28} {rule_found:>6} {total:>6} {overall_acc:>9.1f}% {overall_avg_ms:>10.1f}")
    print(f"{'=' * w}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=None,
                        help="cd into this directory before running")
    parser.add_argument("--delete-investigations", action="store_true",
                        help="Delete investigation files when answer is correct")
    parser.add_argument("--categories", default=None,
                        help="Comma-separated category whitelist")
    args = parser.parse_args()
    cats = args.categories.split(",") if args.categories else None
    run(work_dir=args.work_dir,
        delete_investigations=args.delete_investigations,
        categories=cats)


if __name__ == "__main__":
    main()
