"""Smoke + unit tests for the GSPO / DAPO RL stage runner."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from agent_evolve.backends.tinkerlite.clients.mock import MockTrainingClient
from agent_evolve.training.runners.stages.rl import (
    group_normalize_advantages,
    run_gspo_stage,
)


def test_group_advantages_zscore_within_prompt_group() -> None:
    records = [
        {"pid": 0, "domain": "bits", "correct": True,  "n_tokens": 10},
        {"pid": 0, "domain": "bits", "correct": False, "n_tokens": 12},
        {"pid": 0, "domain": "bits", "correct": True,  "n_tokens": 9},
        {"pid": 0, "domain": "bits", "correct": False, "n_tokens": 11},
    ]
    group_normalize_advantages(records, advantage_mode="group")
    # Mean reward = 0.5 → correct rollouts get +A, incorrect get -A.
    adv = [r["advantage"] for r in records]
    assert adv[0] > 0 and adv[2] > 0
    assert adv[1] < 0 and adv[3] < 0
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-6)


def test_loop_advantage_uses_leave_one_out() -> None:
    records = [
        {"pid": 0, "domain": "bits", "correct": True,  "n_tokens": 10},
        {"pid": 0, "domain": "bits", "correct": False, "n_tokens": 10},
    ]
    group_normalize_advantages(records, advantage_mode="loop")
    # R=[1,0], LOO: A0 = 1 - mean([0]) = 1; A1 = 0 - mean([1]) = -1.
    assert math.isclose(records[0]["advantage"], 1.0)
    assert math.isclose(records[1]["advantage"], -1.0)


def test_length_penalty_reduces_reward_for_long_rollouts() -> None:
    records = [
        {"pid": 0, "domain": "bits", "correct": True, "n_tokens": 2500},
        {"pid": 0, "domain": "bits", "correct": True, "n_tokens": 4500},
    ]
    group_normalize_advantages(
        records,
        advantage_mode="group",
        length_penalty_lambda=1.0,
        length_penalty_cap=2500,
    )
    # Both correct (reward before penalty = 1), but second is 2000 tokens over
    # → reward_combined = 1 - 2.0, advantage is negative.
    assert records[0]["reward_combined"] == 1.0
    assert records[1]["reward_combined"] == pytest.approx(-1.0)
    assert records[0]["advantage"] > records[1]["advantage"]


def test_smoke_gspo_runs_through_training_client(tmp_path: Path) -> None:
    class _FakeWorkspace:
        def __init__(self, root: Path):
            self.root = root

    root = tmp_path / "ws"
    root.mkdir()
    ws = _FakeWorkspace(root)
    client = MockTrainingClient(root)

    stage = {"name": "rl_gspo_smoke", "type": "rl"}
    ckpt, metrics = run_gspo_stage(
        ws,
        stage,
        sampling_client=None,  # unused in smoke
        training_client_factory=lambda: client,
        training_client=client,
        benchmark=None,
        smoke=True,
    )
    assert ckpt.kind == "sampler_weights"
    assert Path(ckpt.path).is_dir()
    assert metrics["stage"] == "rl_gspo_smoke"
    assert metrics["total_rollouts"] == 4
    assert metrics["loss_fn"] == "gspo"
