"""PR1 acceptance: fork creates isolated candidate directory."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.model.types import (
    PatchOperation,
    WorkspaceMutation,
    WorkspacePatch,
)
from agent_evolve.model.workspace import TrainingWorkspace


def _noop_mutation() -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id="m-noop",
        parent_node_id="root",
        description="noop",
        patch=WorkspacePatch(
            operations=[
                PatchOperation(
                    op="replace",
                    path="data/mix.yaml",
                    key_path=["buckets", "default"],
                    value=0.99,
                )
            ]
        ),
        mutation_type="data_mix",
    )


def test_fork_creates_isolated_dir(minimal_workspace: Path, tmp_path: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    work_dir = tmp_path / "work"
    forked = ws.fork("node-a", _noop_mutation(), work_dir=work_dir)

    assert forked.root != ws.root
    assert forked.root.is_dir()
    assert str(work_dir / "nodes" / "node-a" / "workspace") in str(forked.root)
    # Original workspace untouched
    original_mix = ws.read_yaml("data/mix.yaml")
    assert original_mix["buckets"]["default"] == 1.0


def test_fork_excludes_artifact_layers(minimal_workspace: Path, tmp_path: Path) -> None:
    """Artifact layers declared in manifest are not propagated into the fork."""
    # Seed artifacts that must NOT cross the fork boundary.
    (minimal_workspace / "memory" / "records.jsonl").write_text('{"k":"v"}\n')
    (minimal_workspace / "checkpoints" / "adapter.bin").write_bytes(b"stale")
    (minimal_workspace / "evolution" / "incumbent.json").write_text("{}")

    ws = TrainingWorkspace.load(minimal_workspace)
    forked = ws.fork("node-b", _noop_mutation(), work_dir=tmp_path / "work")

    # Artifact directories exist in the fork but are empty.
    for layer in ws.artifact_layers:
        layer_dir = forked.root / layer
        assert layer_dir.is_dir(), f"{layer} should exist in the fork"
        assert list(layer_dir.iterdir()) == [], f"{layer} should be empty in the fork"

    # Evolvable + protected content is copied byte-for-byte.
    assert (forked.root / "data" / "mix.yaml").exists()
    assert (forked.root / "model" / "base.yaml").exists()
    # Original workspace still has the seeded artifact files.
    assert (ws.root / "memory" / "records.jsonl").read_text() == '{"k":"v"}\n'
