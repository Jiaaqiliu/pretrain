"""PR1 acceptance: workspace structural validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_evolve.model.schema import validate_training_workspace
from agent_evolve.model.types import TrainingWorkspaceValidationError
from agent_evolve.model.workspace import TrainingWorkspace


def test_missing_manifest_fails(tmp_path: Path) -> None:
    errors = validate_training_workspace(tmp_path)
    assert any("manifest" in e.lower() for e in errors)


def test_missing_model_base_fails(minimal_workspace: Path) -> None:
    (minimal_workspace / "model" / "base.yaml").unlink()
    errors = validate_training_workspace(minimal_workspace)
    assert any("model/base.yaml" in e for e in errors)


def test_valid_workspace_passes(minimal_workspace: Path) -> None:
    assert validate_training_workspace(minimal_workspace) == []
    ws = TrainingWorkspace.load(minimal_workspace)
    assert ws.name == "fixture_workspace"


def test_load_raises_on_invalid(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(TrainingWorkspaceValidationError):
        TrainingWorkspace.load(tmp_path / "empty")
