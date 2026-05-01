"""End-to-end stage tests using the fake benchmark — solver_distill → data_merge.

No real solvers, no GPU, no benchmark specifics. Validates the pipeline
contract: stage produces correct JSONL + stats, merge dedups + upsamples
+ registers in sources.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_evolve.model.data.base import TrainingExample
from agent_evolve.model.runners.stages.data_merge import run_data_merge_stage
from agent_evolve.model.runners.stages.solver_distill import (
    run_solver_distill_stage,
)

from .fakes import EmptyBenchmark, FakeBenchmark, FakeWorkspace


def _write_recipe(workspace: FakeWorkspace, solver_enabled: bool = True,
                  solver_upsample: int = 2) -> Path:
    recipe_dir = Path(workspace.root) / "data" / "recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipe_dir / "default.yaml"
    recipe_path.write_text(yaml.safe_dump({
        "recipe_name": "test",
        "categories": {
            "fake": {
                "solver": "enabled" if solver_enabled else "disabled",
                "solver_upsample": solver_upsample,
            },
        },
        "filters": {
            "require_verify_pass": True,
            "dedup_by": "prompt_and_source_hash",
        },
    }))
    return recipe_path


def _make_training_rows(n: int = 3) -> list[TrainingExample]:
    return [
        TrainingExample(id=f"row_{i}", prompt=f"prompt {i}",
                        answer=f"answer_{i}", category="fake")
        for i in range(n)
    ]


def test_solver_distill_happy_path(tmp_path: Path) -> None:
    ws = FakeWorkspace(tmp_path)
    _write_recipe(ws)
    rows = _make_training_rows(3)
    benchmark = FakeBenchmark(rows)

    stage = {
        "name": "solver_distill_test",
        "type": "solver_distill",
        "recipe": "data/recipes/default.yaml",
    }
    jsonl_path, stats = run_solver_distill_stage(ws, stage, benchmark=benchmark)

    assert jsonl_path.is_file()
    assert stats["kept"] == 3
    assert stats["per_category_kept"]["fake"] == 3
    assert stats["drop_reasons"] == {}

    # Validate JSONL lines: 3 rows with [verify]: PASS injected and
    # \boxed{<answer>} forced.
    with open(jsonl_path) as f:
        rows_out = [json.loads(l) for l in f]
    assert len(rows_out) == 3
    for r, gt in zip(rows_out, rows):
        assert r["answer"] == gt.answer
        assert f"\\boxed{{{gt.answer}}}" in r["cot"]
        assert "[verify]: PASS" in r["cot"]
        assert r["source"] == "solver"


def test_solver_distill_smoke_mode_produces_deterministic_rows(tmp_path: Path) -> None:
    ws = FakeWorkspace(tmp_path)
    _write_recipe(ws)
    stage = {"name": "smoke", "recipe": "data/recipes/default.yaml"}
    path, stats = run_solver_distill_stage(ws, stage, benchmark=EmptyBenchmark(), smoke=True)
    assert path.is_file()
    assert stats["kept"] == 4
    assert stats["smoke"] is True


def test_solver_distill_empty_benchmark_emits_empty_with_warning(tmp_path: Path) -> None:
    ws = FakeWorkspace(tmp_path)
    _write_recipe(ws)
    stage = {"name": "sd", "recipe": "data/recipes/default.yaml"}
    jsonl_path, stats = run_solver_distill_stage(ws, stage, benchmark=EmptyBenchmark())
    assert jsonl_path.is_file()
    assert jsonl_path.stat().st_size == 0
    assert stats["kept"] == 0
    assert stats["drop_reasons"] == {"no_solvers_registered": 1}


def test_solver_distill_category_disabled_skips(tmp_path: Path) -> None:
    ws = FakeWorkspace(tmp_path)
    _write_recipe(ws, solver_enabled=False)
    benchmark = FakeBenchmark(_make_training_rows(3))
    stage = {"name": "sd", "recipe": "data/recipes/default.yaml"}
    jsonl_path, stats = run_solver_distill_stage(ws, stage, benchmark=benchmark)
    assert stats["kept"] == 0
    assert stats["drop_reasons"]["category_disabled"] == 3


def test_solver_distill_wrong_answer_dropped(tmp_path: Path) -> None:
    ws = FakeWorkspace(tmp_path)
    _write_recipe(ws)
    rows = _make_training_rows(2)
    bench = FakeBenchmark(rows)
    # Sabotage the solver: answer the first prompt wrong
    bench.solvers()   # sanity
    orig_solver = bench.solvers()["fake"]
    wrong_solver = type(orig_solver)(category="fake",
                                      answers={rows[0].prompt: "wrong",
                                               rows[1].prompt: rows[1].answer})
    bench.solvers = lambda: {"fake": wrong_solver}

    stage = {"name": "sd", "recipe": "data/recipes/default.yaml"}
    _, stats = run_solver_distill_stage(ws, stage, benchmark=bench)
    assert stats["kept"] == 1
    assert stats["drop_reasons"]["verifier_failed"] == 1


def test_data_merge_dedup_and_upsample_and_registers(tmp_path: Path) -> None:
    ws = FakeWorkspace(tmp_path)
    _write_recipe(ws, solver_upsample=3)
    rows = _make_training_rows(2)
    benchmark = FakeBenchmark(rows)
    sd_stage = {"name": "sd", "recipe": "data/recipes/default.yaml",
                "out_subdir": "data/generated/sd"}
    run_solver_distill_stage(ws, sd_stage, benchmark=benchmark)

    # Also create a 2nd stage's output manually with a duplicate row + new row
    second_dir = Path(ws.root) / "data" / "generated" / "other"
    second_dir.mkdir(parents=True, exist_ok=True)
    with open(second_dir / "rows.jsonl", "w") as f:
        # Duplicate of the first solver_distill row (same prompt + source)
        # -> should dedup. New prompt -> should survive.
        f.write(json.dumps({
            "id": "dup", "prompt": rows[0].prompt, "answer": rows[0].answer,
            "category": "fake", "cot": "whatever", "source": "solver",
            "metadata": {},
        }) + "\n")
        f.write(json.dumps({
            "id": "new", "prompt": "novel_prompt", "answer": "novel",
            "category": "fake", "cot": "trace", "source": "teacher_llm",
            "metadata": {},
        }) + "\n")

    merge_stage = {
        "name": "merge",
        "recipe": "data/recipes/default.yaml",
        "inputs": ["data/generated/sd", "data/generated/other"],
        "out_subdir": "data/generated/merge",
        "upsample_from_recipe": True,
    }
    merged_path, stats = run_data_merge_stage(ws, merge_stage)

    assert merged_path.is_file()
    assert stats["rows_before_dedup"] == 4   # 2 from sd + 2 from other
    assert stats["rows_after_dedup"] == 3    # 1 duplicate removed
    # Upsample: 2 solver-source @3x + 1 teacher-source @1x (default) = 7
    assert stats["rows_final_after_upsample"] == 7

    # sources.yaml should have exactly one entry pointing at merged.jsonl
    sources = yaml.safe_load((Path(ws.root) / "data" / "sources.yaml").read_text())
    entries = [s for s in sources.get("sources", []) if s.get("path") == str(merged_path)]
    assert len(entries) == 1


def test_data_merge_missing_input_recorded_but_not_fatal(tmp_path: Path) -> None:
    ws = FakeWorkspace(tmp_path)
    _write_recipe(ws)
    stage = {
        "name": "merge",
        "recipe": "data/recipes/default.yaml",
        "inputs": ["data/generated/does_not_exist"],
        "out_subdir": "data/generated/merge",
    }
    merged_path, stats = run_data_merge_stage(ws, stage)
    assert merged_path.is_file()
    assert stats["rows_before_dedup"] == 0
    assert len(stats["input_errors"]) == 1
    assert "missing" in stats["input_errors"][0]["error"]


def test_data_merge_requires_inputs(tmp_path: Path) -> None:
    ws = FakeWorkspace(tmp_path)
    _write_recipe(ws)
    with pytest.raises(ValueError, match="inputs"):
        run_data_merge_stage(ws, {"name": "m", "recipe": "data/recipes/default.yaml"})
