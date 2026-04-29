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

import contextlib
import contextvars
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ...training.types import CheckpointRef
from .common_cfg import (
    _common_config,
    build_gspo_cfg,
    build_sft_cfg,
    default_world_size as _default_world_size,
)


# ── Stage-runner override hook ─────────────────────────────────────────
#
# Default spawn is a local ``torchrun`` subprocess (_spawn_torchrun below).
# An alternate backend (e.g. the k8s elastic scheduler) can swap in a
# different runner via ``override_stage_runner(fn)`` — ``fn`` must ensure
# ``.ddp_result.json`` exists at the cfg's ``out_result_path`` before
# returning. Signature: ``fn(cfg_path, world_size, log_prefix) -> None``.
#
# Scoped via ContextVar so parallel callers (async fan-out) don't bleed
# into each other.

StageRunner = Callable[[Path, int, str], None]
_stage_runner_cv: "contextvars.ContextVar[StageRunner | None]" = contextvars.ContextVar(
    "ae_ddp_stage_runner", default=None,
)


@contextlib.contextmanager
def override_stage_runner(fn: StageRunner):
    """Context manager: route DDP stage spawns through ``fn`` instead of
    the default local torchrun subprocess."""
    token = _stage_runner_cv.set(fn)
    try:
        yield
    finally:
        _stage_runner_cv.reset(token)


def _dispatch_stage(cfg_path: Path, world_size: int, log_prefix: str) -> None:
    runner = _stage_runner_cv.get()
    if runner is not None:
        runner(cfg_path, world_size, log_prefix)
        return
    _spawn_torchrun(cfg_path, world_size, log_prefix)


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

    sft_cfg = build_sft_cfg(
        workspace, stage,
        base_cfg=base_cfg, adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg, batching_cfg=batching_cfg,
        outdir=outdir, result_path=result_path,
        start_adapter_path=start_adapter_path,
        budget_seconds=budget_seconds,
    )
    cfg_path.write_text(json.dumps(sft_cfg, indent=2))
    _dispatch_stage(cfg_path, ws, log_prefix="sft-ddp")
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

    full_cfg = build_gspo_cfg(
        workspace, stage,
        base_cfg=base_cfg, adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg,
        rollouts_path=rollouts_path,
        start_adapter_path=start_adapter_path,
        gspo_cfg=gspo_cfg,
        outdir=outdir, result_path=result_path,
    )
    cfg_path.write_text(json.dumps(full_cfg, indent=2))
    _dispatch_stage(cfg_path, ws, log_prefix="gspo-ddp")
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


__all__ = [
    "run_sft_ddp",
    "run_gspo_ddp",
    "override_stage_runner",
    "StageRunner",
]
