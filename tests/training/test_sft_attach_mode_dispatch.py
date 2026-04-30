"""SFT stage attach-mode dispatch — pure unit test.

`runners/stages/sft.py::_run_real_stage` reads
``model/adapter.yaml::type``, resolves it via the ``ModelAdapter`` registry,
and routes ``ATTACH_MODE_INPLACE`` adapters onto the HF Trainer path
(``_run_real_stage_full_param``) instead of the LoRA step-driven path
(``_run_real_stage`` body).

This test pins that contract without booting torch / HF / DeepSpeed:
we monkey-patch the two implementations and check that the *correct*
one fires for each adapter kind. If someone refactors the dispatch and
breaks routing, this test fails immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_evolve.training.runners.stages import sft as sft_stage
from agent_evolve.training.types import CheckpointRef


class _FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root


def _write_adapter_yaml(root: Path, adapter_type: str | None) -> None:
    (root / "model").mkdir(parents=True, exist_ok=True)
    if adapter_type is None:
        return
    (root / "model" / "adapter.yaml").write_text(f"type: {adapter_type}\n")


def _write_minimal_seed(root: Path) -> None:
    (root / "model").mkdir(parents=True, exist_ok=True)
    (root / "model" / "base.yaml").write_text("path: /fake/model\n")
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "train" / "batching.yaml").write_text(
        "per_device_bs: 1\ngrad_accum: 1\nmax_seq_len: 128\nlog_every: 1\n"
    )
    (root / "train" / "optimizer.yaml").write_text("lr: 1.0e-5\n")


def test_full_deepspeed_adapter_routes_to_full_param_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_minimal_seed(tmp_path)
    _write_adapter_yaml(tmp_path, "full_deepspeed_customized")

    fired: list[str] = []
    fake_ckpt = CheckpointRef(name="x", path=str(tmp_path), kind="full_state")

    def _fake_full_param(ws: Any, stage: dict, opt: dict | None, budget: float | None):
        fired.append("full_param")
        return fake_ckpt, {"stage": stage.get("name")}

    def _fake_ddp(ws: Any, stage: dict, opt: dict | None, budget: float | None):
        fired.append("ddp")
        raise AssertionError("DDP path must NOT fire for ATTACH_MODE_INPLACE")

    monkeypatch.setattr(sft_stage, "_run_real_stage_full_param", _fake_full_param)
    monkeypatch.setattr(sft_stage, "_run_real_stage_ddp", _fake_ddp)
    # Force AE_TRAIN_DDP=1 to confirm full-param path takes priority.
    monkeypatch.setenv("AE_TRAIN_DDP", "1")

    ws = _FakeWorkspace(tmp_path)
    ckpt, _ = sft_stage._run_real_stage(
        ws,
        stage={"name": "sft", "epochs": 1},
        optimizer=None,
        budget_seconds=10.0,
        training_client=None,
    )

    assert fired == ["full_param"], (
        f"expected only the full_param path to fire; got {fired}"
    )
    assert ckpt is fake_ckpt


def test_lora_adapter_does_not_route_to_full_param_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_minimal_seed(tmp_path)
    _write_adapter_yaml(tmp_path, "lora")  # current default

    fired: list[str] = []

    def _fake_full_param(*args, **kwargs):
        fired.append("full_param")
        raise AssertionError("full_param path must NOT fire for LoRA")

    def _fake_ddp(*args, **kwargs):
        fired.append("ddp")
        return CheckpointRef(name="x", path=str(tmp_path), kind="adapter"), {}

    monkeypatch.setattr(sft_stage, "_run_real_stage_full_param", _fake_full_param)
    monkeypatch.setattr(sft_stage, "_run_real_stage_ddp", _fake_ddp)
    monkeypatch.setenv("AE_TRAIN_DDP", "1")  # makes the LoRA path go to DDP fan-out

    ws = _FakeWorkspace(tmp_path)
    sft_stage._run_real_stage(
        ws,
        stage={"name": "sft", "epochs": 1},
        optimizer=None,
        budget_seconds=10.0,
        training_client=None,
    )

    assert fired == ["ddp"], (
        f"expected DDP path (LoRA + AE_TRAIN_DDP=1); got {fired}"
    )


def test_missing_adapter_yaml_falls_back_to_wrap_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``model/adapter.yaml`` → default ``ATTACH_MODE_WRAP`` (LoRA-style)."""
    _write_minimal_seed(tmp_path)
    # Deliberately don't write adapter.yaml.

    from agent_evolve.backends.tinkerlite.adapters import ATTACH_MODE_WRAP

    ws = _FakeWorkspace(tmp_path)
    assert sft_stage._resolve_attach_mode(ws) == ATTACH_MODE_WRAP


def test_unknown_adapter_type_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Misconfigured workspace surfaces as a loud KeyError, not a silent
    fallback into the LoRA path. Catches typos like
    ``type: full_deepspeed_custom`` instead of ``full_deepspeed_customized``."""
    _write_minimal_seed(tmp_path)
    _write_adapter_yaml(tmp_path, "no_such_adapter_kind")

    ws = _FakeWorkspace(tmp_path)
    with pytest.raises(KeyError, match="Unknown adapter kind"):
        sft_stage._resolve_attach_mode(ws)
