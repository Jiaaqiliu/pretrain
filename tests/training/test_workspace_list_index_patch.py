"""List-index patch support: key_path can have int indices for YAML lists."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_evolve.training.types import (
    PatchOperation,
    WorkspaceMutation,
    WorkspacePatch,
)
from agent_evolve.training.workspace import TrainingWorkspace


def _flip_stage_enabled(index: int, value: bool) -> WorkspaceMutation:
    return WorkspaceMutation(
        mutation_id="m-stage",
        parent_node_id="root",
        description="flip stage enabled",
        patch=WorkspacePatch(operations=[
            PatchOperation(
                op="replace",
                path="train/pipeline.yaml",
                key_path=["stages", index, "enabled"],
                value=value,
            ),
        ]),
        mutation_type="training_recipe",
    )


def test_list_index_key_path_flips_stage_enabled(
    minimal_workspace: Path, tmp_path: Path
) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    # Seed workspace already has one sft_warmup stage with enabled=True in the
    # fixture; flip it to False via list-index patch.
    forked = ws.fork("nx", _flip_stage_enabled(0, False), work_dir=tmp_path / "w")

    with open(forked.root / "train" / "pipeline.yaml") as f:
        pipeline = yaml.safe_load(f)

    # The stages list must still be a list (not overwritten with a dict) and
    # stages[0].enabled must be False.
    assert isinstance(pipeline["stages"], list)
    assert pipeline["stages"][0]["enabled"] is False
    # Other fields in stages[0] must be preserved.
    assert pipeline["stages"][0]["name"] == "sft_warmup"
    assert pipeline["stages"][0]["type"] == "sft"
