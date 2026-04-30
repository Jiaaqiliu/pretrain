"""``SolverDistillGenerator`` — thin ``DataGenerator`` around the existing
``runners/stages/solver_distill.py`` worker.

Exists so pipelines can use the new unified ``type: generate, generator:
solver_distill`` syntax in addition to the legacy ``type: solver_distill``
key. No behavior change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..base import GeneratedRow
from ..generator import register_data_generator
from ..recipe import DataRecipe


@register_data_generator("solver_distill")
class SolverDistillGenerator:
    name = "solver_distill"

    def generate(
        self,
        workspace: Any,
        recipe: DataRecipe,
        *,
        benchmark: Any = None,
        budget_seconds: float | None = None,
        smoke: bool = False,
    ) -> Iterable[GeneratedRow]:
        # Delegates to the existing stage worker so logic stays in one place.
        # We mint a transient stage dict so the worker's usual plumbing
        # (out_dir, stats.json) still fires. Callers that want more control
        # can call ``run_solver_distill_stage`` directly.
        from ...runners.stages.solver_distill import run_solver_distill_stage

        stage = {"name": "solver_distill_gen", "recipe_inline": recipe}
        out_path, _stats = run_solver_distill_stage(
            workspace, stage, benchmark=benchmark, smoke=smoke,
        )
        out = Path(out_path)
        if not out.is_file():
            return
        with out.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield GeneratedRow.from_dict(json.loads(line))
