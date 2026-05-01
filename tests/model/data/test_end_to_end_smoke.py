"""End-to-end smoke: drive solver_distill + data_merge through the
real dispatcher in ``SingleNodeTinkerLiteBackend._run_pipeline``.

We skip the `sft`/`rl` stages (they need torch + a real dataset) — the
goal is to prove the dispatch + stage contract works all the way from
``pipeline.yaml`` → worker → FSx artifact. The ``mock=True`` backend
produces a synthetic 4-row smoke output from ``solver_distill`` that
``data_merge`` then picks up, dedups, upsamples, and registers.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend

from .fakes import EmptyBenchmark, FakeWorkspace


def _make_workspace(tmp_path: Path, stages: list[dict]) -> FakeWorkspace:
    """Write a minimal workspace: recipe + pipeline.yaml only. The
    solver_distill smoke path doesn't need a train dataset."""
    ws = FakeWorkspace(tmp_path)
    root = Path(ws.root)

    # Recipe: smoke mode doesn't consult per-category enabled flags,
    # but the loader enforces schema validity so we emit a minimal one.
    recipe_dir = root / "data" / "recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "default.yaml").write_text(yaml.safe_dump({
        "recipe_name": "e2e_smoke",
        "categories": {"smoke": {"solver": "enabled", "solver_upsample": 2}},
        "filters": {"require_verify_pass": True, "dedup_by": "prompt_and_source_hash"},
    }))

    # Pipeline
    pipeline_dir = root / "train"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "pipeline.yaml").write_text(yaml.safe_dump({"stages": stages}))
    return ws


def test_pipeline_dispatch_runs_solver_distill_then_data_merge(tmp_path: Path) -> None:
    stages = [
        {
            "name": "solver_distill",
            "type": "solver_distill",
            "enabled": True,
            "recipe": "data/recipes/default.yaml",
            "out_subdir": "data/generated/solver_distill",
        },
        {
            "name": "data_merge",
            "type": "data_merge",
            "enabled": True,
            "recipe": "data/recipes/default.yaml",
            "inputs": ["data/generated/solver_distill"],
            "out_subdir": "data/generated/data_merge",
            "upsample_from_recipe": True,
            "register_in_sources": True,
        },
    ]
    ws = _make_workspace(tmp_path, stages)
    root = Path(ws.root)

    backend = SingleNodeTinkerLiteBackend(mock=True)
    backend._current_workspace = ws
    backend._current_benchmark = EmptyBenchmark()  # smoke path doesn't need solvers
    backend._current_split = "smoke"

    # Drive the pipeline dispatcher directly — we don't need the full
    # run_trial wrapper (that path calls SFT/RL and evaluation).
    pipeline = yaml.safe_load((root / "train" / "pipeline.yaml").read_text())

    # _run_pipeline expects a budget; give it a trivial one.
    from agent_evolve.model.types import TrialBudget
    try:
        ckpt, metrics = backend._run_pipeline(ws, pipeline, TrialBudget(seconds=60))
    except RuntimeError as exc:
        # _run_pipeline finalizes with pack_adapter when no SFT ran; that's
        # fine — we only care that both data stages ran and wrote artifacts.
        # Surface any other RuntimeError.
        if "pipeline" not in str(exc).lower() and "adapter" not in str(exc).lower():
            raise
        metrics = {"stage_metrics": getattr(backend, "_last_stage_metrics", [])}

    # Verify stage artifacts on disk regardless of finalization outcome.
    sd_jsonl = root / "data" / "generated" / "solver_distill" / "rows.jsonl"
    sd_stats = root / "data" / "generated" / "solver_distill" / "stats.json"
    assert sd_jsonl.is_file(), f"solver_distill JSONL missing: {sd_jsonl}"
    assert sd_stats.is_file()

    merge_jsonl = root / "data" / "generated" / "data_merge" / "merged.jsonl"
    merge_stats = root / "data" / "generated" / "data_merge" / "stats.json"
    assert merge_jsonl.is_file(), f"data_merge output missing: {merge_jsonl}"
    assert merge_stats.is_file()

    # Solver_distill smoke emits 4 rows.
    sd_rows = [json.loads(l) for l in open(sd_jsonl) if l.strip()]
    assert len(sd_rows) == 4
    for r in sd_rows:
        assert r["source"] == "solver"
        assert "\\boxed{" in r["cot"]
        assert "[verify]: PASS" in r["cot"]

    # Data_merge: 4 dedup survivors × solver_upsample=2 => 8 rows.
    merge_rows = [json.loads(l) for l in open(merge_jsonl) if l.strip()]
    merge_stats_dict = json.loads(merge_stats.read_text())
    assert merge_stats_dict["rows_before_dedup"] == 4
    assert merge_stats_dict["rows_after_dedup"] == 4
    assert merge_stats_dict["rows_final_after_upsample"] == 8
    assert len(merge_rows) == 8

    # sources.yaml now references the merged JSONL.
    sources = yaml.safe_load((root / "data" / "sources.yaml").read_text())
    entries = [s for s in sources["sources"] if s["path"] == str(merge_jsonl)]
    assert len(entries) == 1
    assert entries[0]["format"] == "jsonl_cot"
