"""Pure config builders for DDP training stages.

Shared by the single-node DDP launcher and the k8s backend so both execution
paths generate byte-identical ``.ddp_config.json`` files from the same stage
description. The execution-layer concerns (how to spawn torchrun, where the
process runs) live elsewhere.

Guarantee: ``single_node.ddp_launcher.run_sft_ddp`` and the k8s backend MUST
produce the same dict from the same inputs — this is the only place that
definition lives.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _common_config(
    workspace: Any,
    stage: dict,
    *,
    base_cfg: dict,
    adapter_cfg: dict,
    optimizer_cfg: dict,
) -> dict:
    """Collect the subset of config that's shared between SFT and GSPO."""
    model_path = base_cfg.get("path") or os.environ.get("AE_BASE_MODEL_PATH")
    if not model_path:
        raise RuntimeError(
            "ddp config requires model/base.yaml::path or AE_BASE_MODEL_PATH"
        )
    return {
        "model_path": str(model_path),
        "lora_rank": int(adapter_cfg.get("rank", 16)),
        "lora_alpha": int(adapter_cfg.get("alpha", 32)),
        "lora_dropout": float(adapter_cfg.get("dropout", 0.05)),
        "target_modules": list(
            adapter_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"],
            )
        ),
        "lr": float(optimizer_cfg.get("lr", 5e-5)),
        "workspace_root": str(workspace.root),
        "ae_root": str(Path(__file__).resolve().parents[3]),
    }


def build_sft_cfg(
    workspace: Any,
    stage: dict,
    *,
    base_cfg: dict,
    adapter_cfg: dict,
    optimizer_cfg: dict,
    batching_cfg: dict,
    outdir: Path,
    result_path: Path,
    start_adapter_path: str | None = None,
    budget_seconds: float | None = None,
) -> dict:
    """Produce the SFT ``.ddp_config.json`` payload (sans torchrun concerns)."""
    common = _common_config(
        workspace, stage,
        base_cfg=base_cfg, adapter_cfg=adapter_cfg, optimizer_cfg=optimizer_cfg,
    )
    return {
        **common,
        "kind": "sft",
        "start_adapter_path": start_adapter_path,
        "epochs": int(stage.get("epochs", 2)),
        "max_steps": stage.get("max_steps"),
        "per_device_bs": int(batching_cfg.get("per_device_bs", 1)),
        "grad_accum": int(batching_cfg.get("grad_accum", 8)),
        "max_seq_len": int(batching_cfg.get("max_seq_len", 2560)),
        "seed": int(stage.get("seed", 42)),
        "log_every": int(batching_cfg.get("log_every", 5)),
        "warmup_ratio": float(optimizer_cfg.get("warmup_ratio", 0.03)),
        "out_adapter_dir": str(outdir),
        "out_result_path": str(result_path),
        "budget_seconds": budget_seconds,
    }


def build_gspo_cfg(
    workspace: Any,
    stage: dict,
    *,
    base_cfg: dict,
    adapter_cfg: dict,
    optimizer_cfg: dict,
    rollouts_path: str | Path,
    start_adapter_path: str,
    gspo_cfg: dict,
    outdir: Path,
    result_path: Path,
) -> dict:
    """Produce the GSPO ``.ddp_config.json`` payload."""
    common = _common_config(
        workspace, stage,
        base_cfg=base_cfg, adapter_cfg=adapter_cfg, optimizer_cfg=optimizer_cfg,
    )
    # GSPO LR overrides the optimizer.yaml default.
    common["lr"] = float(gspo_cfg["lr"])
    return {
        **common,
        "kind": "gspo",
        "start_adapter_path": start_adapter_path,
        "rollouts_path": str(rollouts_path),
        "epochs": int(gspo_cfg.get("epochs", 1)),
        "grad_accum": int(gspo_cfg.get("grad_accum", 8)),
        "eps_low": float(gspo_cfg.get("eps_low", 3e-4)),
        "eps_high": float(gspo_cfg.get("eps_high", 4e-4)),
        "dapo_token_level": bool(gspo_cfg.get("dapo_token_level", False)),
        "max_steps": gspo_cfg.get("max_steps"),
        "log_every": int(gspo_cfg.get("log_every", 4)),
        "seed": int(gspo_cfg.get("seed", 11)),
        "out_adapter_dir": str(outdir),
        "out_result_path": str(result_path),
    }


def default_world_size() -> int:
    """Number of GPUs to use for DDP. Honors CUDA_VISIBLE_DEVICES if set."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip()])
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


__all__ = [
    "build_sft_cfg",
    "build_gspo_cfg",
    "default_world_size",
    "_common_config",
]
