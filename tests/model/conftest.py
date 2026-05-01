"""Shared fixtures for training subsystem tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


MINIMAL_MANIFEST = {
    "name": "fixture_workspace",
    "contract_version": "train-1.0",
    "defaults": {
        "benchmark": "nemo_reasoner",
        "algorithm": "mcgs",
        "backend": "h200_single_node",
    },
    "evolvable_layers": [
        "data/mix.yaml",
        "data/curriculum.yaml",
        "train/pipeline.yaml",
    ],
    "protected_layers": [
        "model/base.yaml",
        "eval/local_splits.yaml",
    ],
    "artifact_layers": [
        "memory",
        "checkpoints",
        "evolution",
    ],
}


def _minimal_train_pipeline() -> dict:
    return {
        "stages": [
            {
                "name": "sft_warmup",
                "type": "sft",
                "steps": 2,
                "loss": "cross_entropy",
                "data_mix": "default",
            }
        ]
    }


def _minimal_data_mix() -> dict:
    return {"buckets": {"default": 1.0}}


def _minimal_local_splits() -> dict:
    return {"splits": {"local_holdout_small": "eval/local_holdout_small.jsonl"}}


def _minimal_error_taxonomy() -> dict:
    return {
        "buckets": [
            {"id": "format_error"},
            {"id": "wrong_rule"},
            {"id": "eval_runtime_error"},
        ]
    }


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(value, f, default_flow_style=False, sort_keys=False)


def build_minimal_workspace(root: Path, *, overrides: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(MINIMAL_MANIFEST)
    if overrides:
        manifest.update(overrides)
    _write_yaml(root / "manifest.yaml", manifest)
    _write_yaml(root / "model" / "base.yaml", {"name": "toy-7b"})
    _write_yaml(root / "model" / "adapter.yaml", {"type": "lora", "rank": 8})
    _write_yaml(root / "data" / "sources.yaml", {"sources": []})
    _write_yaml(root / "data" / "mix.yaml", _minimal_data_mix())
    _write_yaml(root / "data" / "curriculum.yaml", {"curriculum": []})
    _write_yaml(root / "train" / "pipeline.yaml", _minimal_train_pipeline())
    _write_yaml(root / "eval" / "local_splits.yaml", _minimal_local_splits())
    _write_yaml(root / "eval" / "error_taxonomy.yaml", _minimal_error_taxonomy())
    for d in ["memory", "checkpoints", "evolution"]:
        (root / d).mkdir(exist_ok=True)
    return root


@pytest.fixture
def minimal_workspace(tmp_path: Path) -> Path:
    """Return a freshly-built minimal training workspace on disk."""
    return build_minimal_workspace(tmp_path / "ws")
