"""PR-D targeted tests: ``DataGenerator`` Protocol + registry."""

from __future__ import annotations

from typing import Any, Iterable

import pytest

from agent_evolve.training.data import (
    DATA_GENERATORS,
    DataGenerator,
    GeneratedRow,
    register_data_generator,
    resolve_data_generator,
)


def test_builtin_generators_registered_on_data_import():
    import agent_evolve.training.data  # noqa: F401

    for name in ("solver_distill", "teacher_llm"):
        assert name in DATA_GENERATORS


def test_resolve_unknown_generator_raises():
    with pytest.raises(KeyError, match="Unknown data generator"):
        resolve_data_generator("not_a_real_generator")


def test_register_new_generator_class():
    @register_data_generator("ut_perturb")
    class _UtPerturb:
        name = "ut_perturb"

        def generate(self, workspace, recipe, **_kw) -> Iterable[GeneratedRow]:
            yield GeneratedRow(
                id="1", prompt="p", answer="a", category="c",
                cot="c", source="ut_perturb",
            )

    try:
        g = resolve_data_generator("ut_perturb")
        assert isinstance(g, DataGenerator)
        rows = list(g.generate(workspace=None, recipe=None))
        assert rows[0].source == "ut_perturb"
    finally:
        DATA_GENERATORS.pop("ut_perturb", None)


def test_generate_stage_is_registered():
    """The new unified ``type: generate`` stage wires to the DataGenerator registry."""
    from agent_evolve.training.stage_registry import STAGE_TYPES

    import agent_evolve.training.runners  # noqa: F401 — triggers

    assert "generate" in STAGE_TYPES


def test_generate_stage_dispatches_to_registered_generator(tmp_path):
    """End-to-end: register a fake generator, run the generate stage, verify
    its rows land in rows.jsonl."""
    from types import SimpleNamespace

    from agent_evolve.training.runners.stages.generate import _generate_stage_adapter
    from agent_evolve.training.stage_registry import StageContext

    @register_data_generator("ut_e2e_gen")
    class _UtE2EGen:
        name = "ut_e2e_gen"

        def generate(self, workspace, recipe, **_kw):
            for i in range(3):
                yield GeneratedRow(
                    id=f"r{i}", prompt="p", answer="a", category="c",
                    cot="cot", source="ut_e2e_gen",
                )

    try:
        ws = SimpleNamespace(root=tmp_path)
        ctx = StageContext(
            workspace=ws,
            stage={
                "name": "ut_e2e_stage", "type": "generate",
                "generator": "ut_e2e_gen", "recipe_inline": object(),  # skips YAML load
            },
            benchmark=None, budget_seconds=None, smoke=True, last_ckpt=None,
        )
        result = _generate_stage_adapter(ctx)
        assert result.checkpoint is None
        assert result.metrics["rows_written"] == 3
        assert result.metrics["generator"] == "ut_e2e_gen"
        rows_jsonl = tmp_path / "data" / "generated" / "ut_e2e_stage" / "rows.jsonl"
        assert rows_jsonl.is_file()
        lines = rows_jsonl.read_text().splitlines()
        assert len(lines) == 3
    finally:
        DATA_GENERATORS.pop("ut_e2e_gen", None)
