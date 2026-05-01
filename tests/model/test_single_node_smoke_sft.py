"""PR7 acceptance: one SFT stage produces a checkpoint."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend
from agent_evolve.model.types import (
    TrainingSearchNode,
    TrialBudget,
)
from agent_evolve.model.workspace import TrainingWorkspace

from .fakes import FakeBenchmark


def test_one_fake_sft_stage_produces_checkpoint(minimal_workspace: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    backend = SingleNodeTinkerLiteBackend(mock=True)
    node = TrainingSearchNode(node_id="n-sft", parent_id=None, branch_id=0)
    result = backend.run_trial(ws, node, TrialBudget(seconds=30), FakeBenchmark())
    assert result.status == "success"
    assert result.checkpoint is not None
    adapter_path = Path(result.checkpoint.path)
    assert adapter_path.is_dir()
    assert (adapter_path / "adapter.json").exists()
