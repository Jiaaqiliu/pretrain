"""``TeacherLLMGenerator`` — thin ``DataGenerator`` around the existing
``runners/stages/teacher_distill.py`` worker.

Exists so pipelines can use ``type: generate, generator: teacher_llm`` in
addition to the legacy ``type: synth_generate`` key. No behavior change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..base import GeneratedRow
from ..generator import register_data_generator
from ..recipe import DataRecipe


@register_data_generator("teacher_llm")
class TeacherLLMGenerator:
    name = "teacher_llm"

    def generate(
        self,
        workspace: Any,
        recipe: DataRecipe,
        *,
        benchmark: Any = None,
        budget_seconds: float | None = None,
        smoke: bool = False,
    ) -> Iterable[GeneratedRow]:
        from ...runners.stages.teacher_distill import run_synth_stage

        stage = {"name": "teacher_llm_gen"}
        out_path, _stats = run_synth_stage(
            workspace, stage, smoke=smoke, budget_seconds=budget_seconds,
        )
        out = Path(out_path)
        if not out.is_file():
            return
        with out.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # The teacher-distill worker writes free-form JSONL (not
                # GeneratedRow schema); normalize here. Anything missing
                # defaults to empty strings / the ``teacher_llm`` source.
                yield GeneratedRow(
                    id=row.get("id", ""),
                    prompt=row.get("prompt", row.get("prompt_rendered", "")),
                    answer=row.get("answer", ""),
                    category=row.get("domain", row.get("category", "")),
                    cot=row.get("completion", row.get("cot", "")),
                    source="teacher_llm",
                    metadata={k: v for k, v in row.items() if k not in
                              {"id", "prompt", "prompt_rendered", "answer",
                               "category", "domain", "cot", "completion"}},
                )
