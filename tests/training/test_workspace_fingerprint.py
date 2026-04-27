"""PR1 acceptance: fingerprint reflects evolvable changes but ignores artifacts."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.training.types import (
    PatchOperation,
    WorkspaceMutation,
    WorkspacePatch,
)
from agent_evolve.training.workspace import TrainingWorkspace


def test_fingerprint_changes_on_evolvable_mutation(
    minimal_workspace: Path, tmp_path: Path
) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    before = ws.fingerprint().evolvable_hash

    mutation = WorkspaceMutation(
        mutation_id="m1",
        parent_node_id="root",
        description="bump",
        patch=WorkspacePatch(
            operations=[
                PatchOperation(
                    op="replace",
                    path="data/mix.yaml",
                    key_path=["buckets", "default"],
                    value=0.75,
                )
            ]
        ),
        mutation_type="data_mix",
    )
    forked = ws.fork("node-x", mutation, work_dir=tmp_path / "work")
    after = forked.fingerprint().evolvable_hash
    assert before != after


def test_fingerprint_stable_for_artifact_writes(minimal_workspace: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    before = ws.fingerprint()
    ws.append_jsonl("evolution/reports/cycle_test.jsonl", {"cycle": 1})
    after = ws.fingerprint()
    assert before.evolvable_hash == after.evolvable_hash
    assert before.protected_hash == after.protected_hash
