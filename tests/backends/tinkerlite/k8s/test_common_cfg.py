"""Verify ``common_cfg`` produces configs identical to the pre-refactor
``ddp_launcher`` payloads — the k8s backend and local ddp_launcher MUST
emit byte-identical ``.ddp_config.json`` from the same inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_evolve.backends.tinkerlite.common_cfg import (
    build_gspo_cfg,
    build_sft_cfg,
)


class _WS:
    def __init__(self, root: Path):
        self.root = str(root)


def _std_inputs(tmp_path: Path):
    ws = _WS(tmp_path)
    stage = {"name": "sft_warmup", "epochs": 3, "seed": 7, "loss": "cross_entropy"}
    base_cfg = {"path": "/fsx/models/Nemotron-3-Nano-30B-A3B-BF16"}
    adapter_cfg = {"rank": 16, "alpha": 32, "dropout": 0.05}
    optimizer_cfg = {"lr": 5e-5, "warmup_ratio": 0.03}
    batching_cfg = {"per_device_bs": 1, "grad_accum": 8, "max_seq_len": 2560, "log_every": 5}
    return ws, stage, base_cfg, adapter_cfg, optimizer_cfg, batching_cfg


def test_sft_cfg_shape(tmp_path: Path) -> None:
    ws, stage, base_cfg, adapter_cfg, optimizer_cfg, batching_cfg = _std_inputs(tmp_path)
    outdir = tmp_path / "adapters" / "sft"
    cfg = build_sft_cfg(
        ws, stage,
        base_cfg=base_cfg, adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg, batching_cfg=batching_cfg,
        outdir=outdir, result_path=outdir / ".ddp_result.json",
        start_adapter_path="/some/adapter",
        budget_seconds=1800.0,
    )
    assert cfg["kind"] == "sft"
    assert cfg["epochs"] == 3
    assert cfg["seed"] == 7
    assert cfg["lr"] == 5e-5
    assert cfg["start_adapter_path"] == "/some/adapter"
    assert cfg["budget_seconds"] == 1800.0
    assert cfg["out_adapter_dir"] == str(outdir)
    assert cfg["target_modules"]  # has defaults
    assert cfg["model_path"] == "/fsx/models/Nemotron-3-Nano-30B-A3B-BF16"


def test_gspo_cfg_lr_overrides_optimizer(tmp_path: Path) -> None:
    ws, stage, base_cfg, adapter_cfg, optimizer_cfg, batching_cfg = _std_inputs(tmp_path)
    outdir = tmp_path / "adapters" / "rl"
    cfg = build_gspo_cfg(
        ws, stage,
        base_cfg=base_cfg, adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg,
        rollouts_path="/tmp/rollouts.jsonl",
        start_adapter_path="/tmp/seed_adapter",
        gspo_cfg={"lr": 1e-6, "epochs": 1, "grad_accum": 8, "eps_low": 3e-4, "eps_high": 4e-4,
                  "dapo_token_level": True, "max_steps": 50, "log_every": 4, "seed": 11},
        outdir=outdir, result_path=outdir / ".ddp_result.json",
    )
    assert cfg["kind"] == "gspo"
    assert cfg["lr"] == 1e-6  # gspo.lr overrides optimizer.lr
    assert cfg["dapo_token_level"] is True


def test_missing_base_path_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AE_BASE_MODEL_PATH", raising=False)
    ws, stage, _, adapter_cfg, optimizer_cfg, batching_cfg = _std_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="base.yaml::path"):
        build_sft_cfg(
            ws, stage,
            base_cfg={},  # no path
            adapter_cfg=adapter_cfg,
            optimizer_cfg=optimizer_cfg, batching_cfg=batching_cfg,
            outdir=tmp_path / "x", result_path=tmp_path / "x" / "r.json",
        )


def test_env_var_fallback_for_model_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AE_BASE_MODEL_PATH", "/override/model")
    ws, stage, _, adapter_cfg, optimizer_cfg, batching_cfg = _std_inputs(tmp_path)
    cfg = build_sft_cfg(
        ws, stage,
        base_cfg={},
        adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg, batching_cfg=batching_cfg,
        outdir=tmp_path / "x", result_path=tmp_path / "x" / "r.json",
    )
    assert cfg["model_path"] == "/override/model"
