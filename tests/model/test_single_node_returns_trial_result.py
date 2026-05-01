"""PR7 acceptance: backend returns a properly shaped TrainingTrialResult."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend
from agent_evolve.model.types import (
    TrainingSearchNode,
    TrainingTrialResult,
    TrialBudget,
)
from agent_evolve.model.workspace import TrainingWorkspace

from .fakes import FakeBenchmark


def test_returns_training_trial_result(minimal_workspace: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    backend = SingleNodeTinkerLiteBackend(mock=True)
    node = TrainingSearchNode(node_id="n-r", parent_id=None, branch_id=0)
    result = backend.run_trial(ws, node, TrialBudget(seconds=30), FakeBenchmark())

    assert isinstance(result, TrainingTrialResult)
    assert result.node_id == "n-r"
    assert result.workspace_path == str(minimal_workspace)
    assert result.status in {"success", "train_failed", "eval_failed", "invalid_adapter"}
    assert "seconds" in result.cost
