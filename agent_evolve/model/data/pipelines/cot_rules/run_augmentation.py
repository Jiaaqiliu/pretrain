"""Generate all augmented training data.

Walks selected augmenters in
:mod:`agent_evolve.model.data.pipelines.cot_rules.augmenters` and
writes one file per problem to ``augmentations/<id>.txt`` with format::

    [category]
    ...
    [prompt]
    ...
    [completion]
    ...

This is a faithful port of huikang's ``augmentation.py``, adapted to our
package layout.

Usage::

    cd <work_dir>          # matching reads ./reasoning/*.txt
    python -m agent_evolve.model.data.pipelines.cot_rules.run_augmentation
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
from pathlib import Path

OUTPUT_DIR = Path("augmentations")

# Canonical run order (spelling first because it loads slowest).
DEFAULT_ORDER = ("spelling", "concatenation", "splitting", "matching", "lstrip")


def run(
    work_dir: Path | str | None = None,
    *,
    augmenters: list[str] | None = None,
    n_problems_overrides: dict[str, int] | None = None,
) -> None:
    """Programmatic entry point.

    ``augmenters`` is the subset to run, in order; ``None`` runs all.
    ``n_problems_overrides`` lets a YAML override the per-augmenter
    constants (e.g. ``{"concatenation": 1500}``).
    """
    prev_cwd = os.getcwd()
    if work_dir is not None:
        os.chdir(work_dir)
    try:
        return _run_in_cwd(augmenters=augmenters,
                           n_problems_overrides=n_problems_overrides or {})
    finally:
        os.chdir(prev_cwd)


def _run_in_cwd(*, augmenters: list[str] | None,
                n_problems_overrides: dict[str, int]) -> None:
    selected = augmenters or list(DEFAULT_ORDER)
    problems: list[dict[str, str]] = []

    for name in selected:
        if name == "spelling":
            try:
                mod = importlib.import_module(".augmenters.spelling", __package__)
            except (FileNotFoundError, ImportError) as exc:
                print(f"[spelling] skipped: {exc}")
                continue
        else:
            mod = importlib.import_module(f".augmenters.{name}", __package__)
        # Override module-level N_PROBLEMS if requested
        if name in n_problems_overrides and hasattr(mod, "N_PROBLEMS"):
            mod.N_PROBLEMS = n_problems_overrides[name]  # type: ignore[attr-defined]
        problems.extend(mod.generate())

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    for p in problems:
        path = OUTPUT_DIR / f"{p['id']}.txt"
        path.write_text(
            f"[category]\n{p['category']}\n[prompt]\n{p['prompt']}\n[completion]\n{p['completion']}\n"
        )

    cats: dict[str, int] = {}
    for p in problems:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"\nWrote {len(problems)} problems to {OUTPUT_DIR}/")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=None,
                        help="cd into this directory before running")
    parser.add_argument("--augmenters", default=None,
                        help="Comma-separated augmenter whitelist "
                             "(spelling,concatenation,splitting,matching,lstrip)")
    args = parser.parse_args()
    augs = args.augmenters.split(",") if args.augmenters else None
    run(work_dir=args.work_dir, augmenters=augs)


if __name__ == "__main__":
    main()
