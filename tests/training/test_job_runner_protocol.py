"""PR-A targeted tests: ``TrainingJobRunner`` Protocol + registry rename."""

from __future__ import annotations

import pytest

from agent_evolve.training import TrainingJobRunner
from agent_evolve.training.registries import (
    TRAINING_BACKENDS,
    TRAINING_JOB_RUNNERS,
    resolve_backend,
    resolve_job_runner,
)


def test_training_job_runners_is_authoritative():
    """``TRAINING_BACKENDS`` is a backward-compat alias that aliases the
    same dict object. Mutating one must be visible in the other (no
    divergence)."""
    assert TRAINING_BACKENDS is TRAINING_JOB_RUNNERS
    assert resolve_backend is resolve_job_runner


def test_single_node_is_training_job_runner():
    """Existing backend satisfies the new Protocol — no implementation change."""
    from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend

    b = SingleNodeTinkerLiteBackend(mock=True)
    assert isinstance(b, TrainingJobRunner)


def test_tinkerlite_backend_refines_training_job_runner():
    """Any concrete ``TinkerLiteBackend`` instance (LLM-specific refinement)
    also satisfies the generic ``TrainingJobRunner`` Protocol — they share
    ``name`` + ``run_trial`` exactly."""
    from agent_evolve.backends.tinkerlite import TinkerLiteBackend
    from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend

    b = SingleNodeTinkerLiteBackend(mock=True)
    assert isinstance(b, TinkerLiteBackend)
    assert isinstance(b, TrainingJobRunner)


def test_duck_typed_runner_satisfies_protocol():
    """A completely non-LLM class only needs ``name`` + ``run_trial``."""
    from agent_evolve.training.types import TrainingSearchNode, TrainingTrialResult, TrialBudget

    class FakeRunner:
        name = "fake"

        def run_trial(self, workspace, node, budget, benchmark):
            return TrainingTrialResult(
                node_id=node.node_id, workspace_path=str(workspace), status="success",
            )

    assert isinstance(FakeRunner(), TrainingJobRunner)


def test_registry_resolves_job_runner():
    cls = resolve_job_runner("h200_single_node")
    assert cls.name == "h200_single_node"


def test_registry_resolve_backend_alias():
    cls = resolve_backend("h200_single_node")
    assert cls.name == "h200_single_node"
