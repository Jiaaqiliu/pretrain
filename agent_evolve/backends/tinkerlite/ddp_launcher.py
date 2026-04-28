"""Parent-side launcher for DDP training stages.

Dispatches ``torchrun --nproc_per_node=N -m
agent_evolve.training.runners.train_worker_ddp`` as a subprocess. The parent
process stays single-threaded; the subprocess spawns ``world_size`` ranks.

Mirrors the pattern already used by ``synth_worker.py`` (subprocess-isolated
120B teacher distill) — subprocess exit frees all CUDA state cleanly.

Public API:
    run_sft_ddp(workspace, stage, optimizer_cfg, batching_cfg, adapter_cfg,
                base_cfg, start_adapter_path=None, world_size=None,
                budget_seconds=None) -> (CheckpointRef, stats_dict)
    run_gspo_ddp(workspace, stage, records, tokenizer, model_path,
                 start_adapter_path, cfg, world_size=None) -> (CheckpointRef, stats_dict)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ...training.types import CheckpointRef


def _default_world_size() -> int:
    """Number of GPUs to use for DDP. Honors CUDA_VISIBLE_DEVICES if set."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip()])
    # Query torch for the physical count without importing torch at module-load.
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


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
            "ddp_launcher requires model/base.yaml::path or AE_BASE_MODEL_PATH"
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


def _spawn_torchrun(cfg_path: Path, world_size: int, log_prefix: str) -> None:
    ae_root = str(Path(__file__).resolve().parents[3])
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("WANDB_DISABLED", "true")
    env["PYTHONPATH"] = ae_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Use the same python interpreter that the parent is running (matches venv).
    py = sys.executable
    cmd = [
        py,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        "--master_addr=127.0.0.1",
        # Pick a port distinct from anything vLLM might be using. Randomize
        # per-stage so concurrent cycles (if ever) don't collide.
        f"--master_port={29500 + (hash(str(cfg_path)) % 1000)}",
        "-m",
        "agent_evolve.training.runners.train_worker_ddp",
        "--config",
        str(cfg_path),
    ]
    print(f"[{log_prefix}] launching torchrun (world_size={world_size}):", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


# ── Public API ──────────────────────────────────────────────────────────

def run_sft_ddp(
    workspace: Any,
    stage: dict,
    *,
    base_cfg: dict,
    adapter_cfg: dict,
    optimizer_cfg: dict,
    batching_cfg: dict,
    start_adapter_path: str | None = None,
    world_size: int | None = None,
    budget_seconds: float | None = None,
) -> tuple[CheckpointRef, dict]:
    """Launch a DDP SFT stage. Returns (adapter_ref, stats)."""
    ws = world_size or _default_world_size()
    outdir = Path(workspace.root) / "checkpoints" / "adapters" / stage.get("name", "sft")
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / ".ddp_result.json"
    cfg_path = outdir / ".ddp_config.json"

    common = _common_config(
        workspace, stage, base_cfg=base_cfg, adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg,
    )
    sft_cfg = {
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
    cfg_path.write_text(json.dumps(sft_cfg, indent=2))
    _spawn_torchrun(cfg_path, ws, log_prefix="sft-ddp")
    if not result_path.is_file():
        raise RuntimeError(f"DDP worker did not emit {result_path}")
    result = json.loads(result_path.read_text())
    ckpt = CheckpointRef(
        name=stage.get("name", "sft"),
        path=str(outdir),
        kind="adapter",
        metadata={"lr": sft_cfg["lr"], "rank": sft_cfg["lora_rank"], "world_size": ws},
    )
    return ckpt, {
        "stage": stage.get("name"),
        "loss_fn": stage.get("loss", "cross_entropy"),
        **result,
    }


def run_gspo_ddp(
    workspace: Any,
    stage: dict,
    *,
    base_cfg: dict,
    adapter_cfg: dict,
    optimizer_cfg: dict,
    rollouts_path: str | Path,
    start_adapter_path: str,
    gspo_cfg: dict,
    world_size: int | None = None,
) -> tuple[CheckpointRef, dict]:
    """Launch a DDP GSPO update stage on pre-computed rollouts.

    ``gspo_cfg`` should carry: epochs, grad_accum, lr, eps_low, eps_high,
    dapo_token_level, max_steps, log_every, seed.
    """
    ws = world_size or _default_world_size()
    outdir = Path(workspace.root) / "checkpoints" / "adapters" / stage.get("name", "rl_gspo")
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / ".ddp_result.json"
    cfg_path = outdir / ".ddp_config.json"

    common = _common_config(
        workspace, stage, base_cfg=base_cfg, adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg,
    )
    # GSPO LR overrides the optimizer.yaml default.
    common["lr"] = float(gspo_cfg["lr"])

    full_cfg = {
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
    cfg_path.write_text(json.dumps(full_cfg, indent=2))
    _spawn_torchrun(cfg_path, ws, log_prefix="gspo-ddp")
    if not result_path.is_file():
        raise RuntimeError(f"DDP worker did not emit {result_path}")
    result = json.loads(result_path.read_text())
    ckpt = CheckpointRef(
        name=stage.get("name", "rl_gspo"),
        path=str(outdir),
        kind="adapter",
        metadata={"lr": full_cfg["lr"], "world_size": ws},
    )
    return ckpt, {
        "stage": stage.get("name"),
        "loss_fn": "dapo_token_level" if full_cfg["dapo_token_level"] else "gspo",
        **result,
    }


__all__ = ["run_sft_ddp", "run_gspo_ddp"]
