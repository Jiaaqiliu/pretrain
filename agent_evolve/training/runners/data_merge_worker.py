"""``data_merge`` stage worker — dedup across inputs, apply per-category
upsample ratios, register the merged JSONL in ``data/sources.yaml``.

Inputs are directories produced by earlier data-stages, each containing a
``rows.jsonl`` conforming to ``GeneratedRow`` schema. The merged output
is a single ``merged.jsonl`` + ``stats.json`` under
``data/generated/<stage_name>/``. The SFT stage downstream reads
``data/sources.yaml`` as it does today — merged.jsonl shows up as just
another source entry with ``format: jsonl_cot`` so the data loader can
dispatch on format.

Input order matters: a row present in multiple inputs is kept from the
*first* input that contains it (after dedup). That's why the recipe
convention is "solver_distill first, teacher_distill second" — solver
rows are more trustworthy.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ..data.base import GeneratedRow
from ..data.dedup import dedup, upsample
from ..data.recipe import DataRecipe, load_recipe

logger = logging.getLogger(__name__)


# ── Public entry ─────────────────────────────────────────────────────────

def run_data_merge_stage(
    workspace: Any,
    stage: dict,
) -> tuple[Path, dict[str, Any]]:
    """Run the merge stage. Returns (merged.jsonl path, stats dict)."""
    stage_name = stage.get("name", "data_merge")
    out_subdir = stage.get("out_subdir") or f"data/generated/{stage_name}"
    out_dir = Path(workspace.root) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "merged.jsonl"
    stats_path = out_dir / "stats.json"

    inputs = list(stage.get("inputs") or [])
    if not inputs:
        raise ValueError(
            "data_merge stage requires `inputs: [<subdir>, ...]` "
            "pointing at earlier data-stage output directories."
        )

    recipe = _resolve_recipe(workspace, stage)

    # Pass 1: load all inputs in order
    all_rows: list[GeneratedRow] = []
    per_input_counts: list[tuple[str, int]] = []
    per_input_errors: list[tuple[str, str]] = []
    for rel in inputs:
        rows, err = _load_jsonl(Path(workspace.root) / rel / "rows.jsonl")
        if err:
            per_input_errors.append((rel, err))
        all_rows.extend(rows)
        per_input_counts.append((rel, len(rows)))

    # Pass 2: dedup (stable, first-seen-wins)
    before_dedup = len(all_rows)
    deduped = dedup(all_rows, recipe.filters)
    after_dedup = len(deduped)

    # Pass 3: upsample
    final = upsample(deduped, recipe) if stage.get("upsample_from_recipe", True) else deduped

    # Write merged JSONL
    _write_jsonl(merged_path, final)

    # Register in sources.yaml (idempotent)
    if stage.get("register_in_sources", True):
        _append_to_sources(workspace, merged_path, fmt=stage.get("source_format", "jsonl_cot"))

    # Stats
    per_category_final: Counter = Counter()
    per_source_final: Counter = Counter()
    for r in final:
        per_category_final[r.category] += 1
        per_source_final[r.source] += 1

    stats = {
        "stage": stage_name,
        "recipe_name": recipe.recipe_name,
        "inputs": [{"subdir": rel, "rows": n} for rel, n in per_input_counts],
        "input_errors": [{"subdir": rel, "error": err} for rel, err in per_input_errors],
        "rows_before_dedup": before_dedup,
        "rows_after_dedup": after_dedup,
        "rows_final_after_upsample": len(final),
        "per_category": dict(per_category_final),
        "per_source": dict(per_source_final),
        "sources_yaml_updated": bool(stage.get("register_in_sources", True)),
        "dedup_mode": recipe.filters.dedup_by,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return merged_path, stats


# ── Helpers ──────────────────────────────────────────────────────────────

def _resolve_recipe(workspace: Any, stage: dict) -> DataRecipe:
    recipe_rel = stage.get("recipe") or "data/recipes/default.yaml"
    recipe_path = Path(workspace.root) / recipe_rel
    if not recipe_path.is_file():
        raise FileNotFoundError(
            f"data_merge stage expects recipe at {recipe_path} "
            "(set `recipe: <relative path>` on the stage to override)."
        )
    return load_recipe(recipe_path)


def _load_jsonl(path: Path) -> tuple[list[GeneratedRow], str | None]:
    """Return (rows, error_msg). Non-fatal: missing files produce an
    empty list + error string so the merge stage can emit partial output
    rather than blowing up an entire trial."""
    if not path.is_file():
        return [], f"missing file: {path}"
    rows: list[GeneratedRow] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rows.append(GeneratedRow.from_dict(d))
            except (json.JSONDecodeError, KeyError) as exc:
                return rows, f"bad JSONL at {path}:{line_no}: {exc!r}"
    return rows, None


def _write_jsonl(path: Path, rows: list[GeneratedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")


def _append_to_sources(workspace: Any, jsonl_path: Path, *, fmt: str) -> None:
    sources_path = Path(workspace.root) / "data" / "sources.yaml"
    if sources_path.exists():
        raw = yaml.safe_load(sources_path.read_text()) or {}
    else:
        raw = {}
    sources = list(raw.get("sources") or [])
    path_str = str(jsonl_path)
    if any(isinstance(s, dict) and s.get("path") == path_str for s in sources):
        return
    sources.append({"path": path_str, "split": "train", "format": fmt})
    raw["sources"] = sources
    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(yaml.safe_dump(raw, default_flow_style=False, sort_keys=False))


__all__ = ["run_data_merge_stage"]
