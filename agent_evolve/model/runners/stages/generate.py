"""Unified ``type: generate, generator: <name>`` stage.

Replaces the ``type: solver_distill`` / ``type: synth_generate`` pattern with
a generic dispatcher keyed by the ``generator:`` field. Lets user plugins
add new data-generation methods without touching any stage worker.

Legacy ``type: solver_distill`` + ``type: synth_generate`` keys continue to
work (each has its own ``@register_stage`` adapter on the original worker).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ...data.generator import resolve_data_generator
from ...data.recipe import DataRecipe, load_recipe
from ...stage_registry import StageContext, StageResult, register_stage

logger = logging.getLogger(__name__)


@register_stage("generate")
def _generate_stage_adapter(ctx: StageContext) -> StageResult:
    stage = ctx.stage
    gen_name = stage.get("generator")
    if not gen_name:
        raise ValueError(
            f"'generate' stage {stage.get('name')!r} requires a 'generator:' field "
            "pointing at a registered DataGenerator name."
        )
    generator = resolve_data_generator(gen_name)

    recipe = _resolve_recipe(ctx.workspace, stage)

    out_subdir = stage.get("out_subdir") or f"data/generated/{stage.get('name', gen_name)}"
    out_dir = Path(ctx.workspace.root) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rows.jsonl"
    stats_path = out_dir / "stats.json"

    rows_written = 0
    with jsonl_path.open("w") as f:
        for row in generator.generate(
            ctx.workspace,
            recipe,
            benchmark=ctx.benchmark,
            budget_seconds=ctx.budget_seconds,
            smoke=ctx.smoke,
        ):
            f.write(json.dumps(row.to_dict()) + "\n")
            rows_written += 1

    stats = {"rows_written": rows_written, "generator": gen_name}
    stats_path.write_text(json.dumps(stats, indent=2))
    return StageResult(
        checkpoint=None,
        metrics={
            "type": "generate",
            "generator": gen_name,
            "out_path": str(jsonl_path),
            "rows_written": rows_written,
        },
    )


def _resolve_recipe(workspace: Any, stage: dict) -> DataRecipe:
    """Reuse solver_distill's recipe resolution: allow inline ``recipe_inline``
    (for programmatic callers), explicit ``recipe: <relpath>``, or fall back
    to ``data/recipes/default.yaml``."""
    inline = stage.get("recipe_inline")
    if inline is not None:
        return inline
    rel = stage.get("recipe", "data/recipes/default.yaml")
    return load_recipe(Path(workspace.root) / rel)
