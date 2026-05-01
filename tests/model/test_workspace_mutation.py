"""PR1 acceptance: mutations cannot touch protected layers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_evolve.model.types import (
    PatchOperation,
    TrainingWorkspaceValidationError,
    WorkspaceMutation,
    WorkspacePatch,
)
from agent_evolve.model.workspace import TrainingWorkspace


def _make_patch(rel: str, value) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id="m-1",
        parent_node_id="node-root",
        description="test",
        patch=WorkspacePatch(
            operations=[PatchOperation(op="replace", path=rel, value=value)]
        ),
        mutation_type="debug",
    )


def test_mutation_on_protected_fails(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    mutation = _make_patch("model/base.yaml", {"name": "other"})
    with pytest.raises(TrainingWorkspaceValidationError):
        ws.fork("node-1", mutation, work_dir=tmp_path / "work")


def test_mutation_on_evolvable_succeeds(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    mutation = WorkspaceMutation(
        mutation_id="m-2",
        parent_node_id="node-root",
        description="bump failure replay",
        patch=WorkspacePatch(
            operations=[
                PatchOperation(
                    op="replace",
                    path="data/mix.yaml",
                    key_path=["buckets", "failure_replay"],
                    value=0.25,
                )
            ]
        ),
        mutation_type="data_mix",
    )
    forked = ws.fork("node-2", mutation, work_dir=tmp_path / "work")
    mix = forked.read_yaml("data/mix.yaml")
    assert mix["buckets"]["failure_replay"] == 0.25
