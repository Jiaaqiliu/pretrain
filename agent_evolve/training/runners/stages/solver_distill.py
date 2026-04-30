"""``solver_distill`` stage worker — run each category's solver on the
benchmark's training rows, filter by verifier, render CoT, emit JSONL.

Contract (matches synth_worker shape so the dispatcher in single_node
can treat all data-stages uniformly):

    run_solver_distill_stage(workspace, stage, *, benchmark, smoke=False)
        -> (jsonl_path, stats_dict)

Where ``stage`` is the entry from ``train/pipeline.yaml`` and looks like:

    - name: solver_distill_v1
      type: solver_distill
      recipe: data/recipes/default.yaml
      out_subdir: data/generated/solver_distill    # optional

``benchmark`` must implement the optional Protocol additions in
``benchmarks/training_base.py`` — ``.iter_training_rows``, ``.solvers``,
``.verifiers``, ``.cot_renderers``. If any are missing, the worker emits
an empty JSONL and logs why in stats so callers fail loud-but-safe
instead of silently training on garbage.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from ...data.base import GeneratedRow, SolverResult
from ...data.cot_template import CoTPostprocessConfig, postprocess_cot
from ...data.recipe import DataRecipe, load_recipe

logger = logging.getLogger(__name__)


# ── Public entry ─────────────────────────────────────────────────────────

def run_solver_distill_stage(
    workspace: Any,
    stage: dict,
    *,
    benchmark: Any,
    smoke: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run solver_distill. Returns (rows.jsonl path, stats dict)."""
    stage_name = stage.get("name", "solver_distill")
    out_subdir = stage.get("out_subdir") or f"data/generated/{stage_name}"
    out_dir = Path(workspace.root) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rows.jsonl"
    stats_path = out_dir / "stats.json"

    recipe = _resolve_recipe(workspace, stage)

    if smoke:
        return _run_smoke(jsonl_path, stats_path, recipe, stage_name)

    return _run_real(
        workspace=workspace,
        benchmark=benchmark,
        recipe=recipe,
        jsonl_path=jsonl_path,
        stats_path=stats_path,
        stage_name=stage_name,
    )


# ── Smoke path (no benchmark required) ───────────────────────────────────

def _run_smoke(
    jsonl_path: Path,
    stats_path: Path,
    recipe: DataRecipe,
    stage_name: str,
) -> tuple[Path, dict[str, Any]]:
    """Deterministic 4-row output so unit tests and CPU smoke paths have
    a working artifact without needing a real benchmark."""
    rows = [
        GeneratedRow(
            id=f"smoke-{i}",
            prompt=f"smoke prompt {i}\n",
            answer=f"answer-{i}",
            category="smoke",
            cot=postprocess_cot(
                f"Let me think about example {i}.",
                f"answer-{i}",
            ),
            source="solver",
            metadata={"method": "smoke", "confidence": "high"},
        )
        for i in range(4)
    ]
    _write_jsonl(jsonl_path, rows)
    stats = {
        "stage": stage_name,
        "recipe_name": recipe.recipe_name,
        "smoke": True,
        "total_tried": len(rows),
        "kept": len(rows),
        "per_category_tried": {"smoke": len(rows)},
        "per_category_kept": {"smoke": len(rows)},
        "per_category_dropped": {},
        "drop_reasons": {},
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return jsonl_path, stats


# ── Real path ────────────────────────────────────────────────────────────

def _run_real(
    *,
    workspace: Any,
    benchmark: Any,
    recipe: DataRecipe,
    jsonl_path: Path,
    stats_path: Path,
    stage_name: str,
) -> tuple[Path, dict[str, Any]]:
    solvers = _get_optional(benchmark, "solvers", default={})
    verifiers = _get_optional(benchmark, "verifiers", default={})
    renderers = _get_optional(benchmark, "cot_renderers", default={})

    per_cat_tried: Counter = Counter()
    per_cat_kept: Counter = Counter()
    per_cat_dropped: Counter = Counter()
    drop_reasons: Counter = Counter()

    if not solvers:
        stats = {
            "stage": stage_name,
            "recipe_name": recipe.recipe_name,
            "total_tried": 0,
            "kept": 0,
            "drop_reasons": {"no_solvers_registered": 1},
            "message": (
                "benchmark.solvers() returned {}; implement the optional "
                "solver/verifier/cot_renderer Protocols on the benchmark or "
                "disable the solver_distill stage."
            ),
        }
        _write_jsonl(jsonl_path, [])
        stats_path.write_text(json.dumps(stats, indent=2))
        logger.warning("[solver_distill] %s", stats["message"])
        return jsonl_path, stats

    if not hasattr(benchmark, "iter_training_rows"):
        raise RuntimeError(
            "solver_distill requires benchmark.iter_training_rows(workspace) "
            "-- benchmark "
            f"{type(benchmark).__name__} doesn't implement it."
        )

    kept: list[GeneratedRow] = []

    for row in benchmark.iter_training_rows(workspace):
        cat = row.category
        cat_recipe = recipe.category(cat)
        if not cat_recipe.solver_enabled:
            drop_reasons["category_disabled"] += 1
            continue

        solver = solvers.get(cat)
        if solver is None:
            drop_reasons["no_solver_for_category"] += 1
            continue
        verifier = verifiers.get(cat)
        if verifier is None:
            drop_reasons["no_verifier_for_category"] += 1
            continue
        renderer = renderers.get(cat)
        if renderer is None:
            drop_reasons["no_renderer_for_category"] += 1
            continue

        per_cat_tried[cat] += 1

        try:
            result: SolverResult = solver.solve(row.prompt)
        except Exception as exc:  # defensive — a bad solver shouldn't kill the stage
            logger.exception("[solver_distill] solver %s crashed on row %s",
                             cat, row.id)
            drop_reasons["solver_exception"] += 1
            per_cat_dropped[cat] += 1
            continue

        if result.predicted_answer is None or result.confidence == "none":
            drop_reasons["solver_declined"] += 1
            per_cat_dropped[cat] += 1
            continue
        if not verifier.check(result.predicted_answer, row.answer):
            drop_reasons["verifier_failed"] += 1
            per_cat_dropped[cat] += 1
            continue

        try:
            raw_cot = renderer.render(row.prompt, row.answer, result.trace_dict)
        except Exception:
            logger.exception("[solver_distill] renderer %s crashed on row %s",
                             cat, row.id)
            drop_reasons["renderer_exception"] += 1
            per_cat_dropped[cat] += 1
            continue

        cot = postprocess_cot(
            raw_cot,
            row.answer,
            config=CoTPostprocessConfig(
                inject_verify_marker=recipe.filters.require_verify_pass,
                force_boxed_answer=True,
                strip_trailing_whitespace=True,
            ),
        )

        kept.append(GeneratedRow(
            id=row.id,
            prompt=row.prompt,
            answer=row.answer,
            category=cat,
            cot=cot,
            source="solver",
            metadata={
                "method": result.method,
                "confidence": result.confidence,
                **row.metadata,
            },
        ))
        per_cat_kept[cat] += 1

    _write_jsonl(jsonl_path, kept)
    stats = {
        "stage": stage_name,
        "recipe_name": recipe.recipe_name,
        "total_tried": int(sum(per_cat_tried.values())),
        "kept": len(kept),
        "per_category_tried": dict(per_cat_tried),
        "per_category_kept": dict(per_cat_kept),
        "per_category_dropped": dict(per_cat_dropped),
        "drop_reasons": dict(drop_reasons),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return jsonl_path, stats


# ── Helpers ──────────────────────────────────────────────────────────────

def _resolve_recipe(workspace: Any, stage: dict) -> DataRecipe:
    recipe_rel = stage.get("recipe") or "data/recipes/default.yaml"
    recipe_path = Path(workspace.root) / recipe_rel
    if not recipe_path.is_file():
        raise FileNotFoundError(
            f"solver_distill stage expects recipe at {recipe_path} "
            "(set `recipe: <relative path>` on the stage to override)."
        )
    return load_recipe(recipe_path)


def _get_optional(benchmark: Any, attr: str, *, default: Any) -> Any:
    """Call ``benchmark.<attr>()`` if the method exists; fall back to default."""
    method = getattr(benchmark, attr, None)
    if method is None:
        return default
    try:
        return method()
    except NotImplementedError:
        return default


def _write_jsonl(path: Path, rows: list[GeneratedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")


__all__ = ["run_solver_distill_stage"]


# ── StageRegistry adapter ────────────────────────────────────────────────

from ...stage_registry import StageContext, StageResult, register_stage  # noqa: E402


@register_stage("solver_distill")
def _solver_distill_stage_adapter(ctx: StageContext) -> StageResult:
    out_path, stats = run_solver_distill_stage(
        ctx.workspace, ctx.stage, benchmark=ctx.benchmark, smoke=ctx.smoke,
    )
    return StageResult(
        checkpoint=None,
        metrics={"type": "solver_distill", "out_path": str(out_path), **stats},
    )
