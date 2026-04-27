"""PR2 invariant: backend does not compute reward or pick incumbent."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend
from agent_evolve.training.types import (
    TrainingSearchNode,
    TrainingTrialResult,
    TrialBudget,
)
from agent_evolve.training.workspace import TrainingWorkspace

from .fakes import FakeBenchmark


def test_backend_returns_no_reward_field(minimal_workspace: Path) -> None:
    ws = TrainingWorkspace.load(minimal_workspace)
    backend = SingleNodeTinkerLiteBackend(mock=True)
    result = backend.run_trial(
        ws,
        TrainingSearchNode(node_id="n1", parent_id=None, branch_id=0),
        TrialBudget(),
        FakeBenchmark(),
    )
    # TrainingTrialResult has no reward or incumbent fields.
    fields = {f.name for f in TrainingTrialResult.__dataclass_fields__.values()}
    assert "reward" not in fields
    assert "incumbent" not in fields
    assert "is_incumbent" not in fields
    assert "promotion" not in fields
    # And the instance does not carry them either.
    assert not hasattr(result, "reward")
    assert not hasattr(result, "incumbent")


def test_backend_does_not_expose_promotion_api() -> None:
    backend = SingleNodeTinkerLiteBackend(mock=True)
    assert not hasattr(backend, "promote_incumbent")
    assert not hasattr(backend, "select_incumbent")
    assert not hasattr(backend, "compute_reward")
