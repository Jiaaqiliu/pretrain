"""cot_rules pipeline driver.

Reads ``pipeline.yaml``, walks the three steps in order:

  1. step_1_reasoning      → run_reasoning.run(work_dir, ...)
  2. step_2_augmentation   → run_augmentation.run(work_dir, ...)
  3. step_3_corpus         → run_corpus.run(work_dir, ...)

Each step's enabled flag, category subset, augmenter overrides, and
tokenizer choices come from the YAML.

Usage::

    python -m agent_evolve.model.data.pipelines.cot_rules.run \\
        --config agent_evolve/model/data/pipelines/cot_rules/pipeline.yaml \\
        [--from-step 1] [--to-step 3]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

# Steps are imported lazily inside main() so that step 1 can be exercised
# without ``tokenizers`` / ``transformers`` installed (those are only
# required for step 3).


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _selected_augmenters(cfg: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    augs = cfg.get("step_2_augmentation", {}).get("augmenters", {}) or {}
    selected: list[str] = []
    overrides: dict[str, int] = {}
    # canonical order from huikang
    for name in ("spelling", "concatenation", "splitting", "matching", "lstrip"):
        spec = augs.get(name, {})
        if not spec.get("enabled", True):
            continue
        selected.append(name)
        n = spec.get("n_problems")
        if n is not None:
            overrides[name] = int(n)
    return selected, overrides


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--from-step", type=int, default=1, help="1, 2, or 3")
    ap.add_argument("--to-step",   type=int, default=3, help="1, 2, or 3")
    args = ap.parse_args(argv)

    cfg = _load_yaml(args.config)
    paths = cfg["paths"]
    work_dir = Path(paths["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"== cot_rules pipeline: {cfg.get('name','(unnamed)')}")
    print(f"   work_dir = {work_dir}")

    # Step 1 — reasoning (stdlib only)
    if args.from_step <= 1 <= args.to_step:
        s1 = cfg.get("step_1_reasoning", {})
        if s1.get("enabled", True):
            print("\n-- step 1: reasoning")
            from . import run_reasoning
            run_reasoning.run(
                work_dir=work_dir,
                delete_investigations=bool(s1.get("delete_investigations", False)),
                categories=s1.get("categories"),
            )
        else:
            print("\n-- step 1: SKIPPED (disabled in YAML)")

    # Step 2 — augmentation (tokenizers needed only if spelling enabled)
    if args.from_step <= 2 <= args.to_step:
        s2 = cfg.get("step_2_augmentation", {})
        if s2.get("enabled", True):
            print("\n-- step 2: augmentation")
            from . import run_augmentation
            selected, overrides = _selected_augmenters(cfg)
            run_augmentation.run(
                work_dir=work_dir,
                augmenters=selected,
                n_problems_overrides=overrides,
            )
        else:
            print("\n-- step 2: SKIPPED (disabled in YAML)")

    # Step 3 — corpus (requires tokenizers + transformers)
    if args.from_step <= 3 <= args.to_step:
        s3 = cfg.get("step_3_corpus", {})
        if s3.get("enabled", True):
            print("\n-- step 3: corpus")
            from . import run_corpus
            run_corpus.run(
                work_dir=work_dir,
                tokenizer_path=paths.get("tokenizer_path"),
                chat_tokenizer_name=s3.get("chat_tokenizer"),
                train_csv=paths.get("train_csv"),
                token_limit=s3.get("token_limit"),
                prompt_suffix=s3.get("prompt_suffix"),
            )
        else:
            print("\n-- step 3: SKIPPED (disabled in YAML)")

    print("\ncot_rules pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
