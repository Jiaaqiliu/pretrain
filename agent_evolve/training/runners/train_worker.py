"""SFT training worker.

Two paths:

* **smoke** — uses :class:`MockTrainingClient`, no torch. Used by PR8 seed and
  unit tests.
* **real** — drives an :class:`HFTrainingClient` via the
  ``TrainingClient`` protocol (``forward_backward("cross_entropy") +
  optim_step()``). Matches the verified recipe in
  ``../nemotron-auto-research/scripts/train_sft_lora.py`` without going
  through :class:`transformers.Trainer`.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from ...backends.tinkerlite.base import AdamParams, Datum, ModelInput
from ...backends.tinkerlite.hf_clients import (
    HFTrainingClient,
    build_hf_client_from_workspace,
)
from ...backends.tinkerlite.mock_clients import MockTrainingClient
from ..types import CheckpointRef


# ── Public entrypoint ────────────────────────────────────────────────────

def run_sft_stage(
    workspace: Any,
    stage: dict,
    datums: Iterable[Datum] | None = None,
    *,
    optimizer: dict | None = None,
    smoke: bool = True,
    budget_seconds: float | None = None,
    training_client: HFTrainingClient | None = None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Run one SFT stage. Returns the checkpoint plus a metrics dict.

    ``training_client`` is optional: if provided, the real path reuses it (so
    the model is loaded once across multiple stages in the same trial). If
    omitted, the real path constructs one from workspace YAML.
    """
    if smoke:
        return _run_smoke_stage(workspace, stage, datums or [], optimizer, budget_seconds)
    return _run_real_stage(
        workspace,
        stage,
        optimizer,
        budget_seconds,
        training_client=training_client,
    )


# ── Smoke path (no torch) ────────────────────────────────────────────────

def _run_smoke_stage(
    workspace: Any,
    stage: dict,
    datums: Iterable[Datum],
    optimizer: dict | None,
    budget_seconds: float | None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    client = MockTrainingClient(Path(workspace.root))
    loss_fn = stage.get("loss", "cross_entropy")
    lr = (optimizer or {}).get("lr", 1e-4)
    total_loss = 0.0
    total_steps = 0
    start = time.time()
    batch = list(datums) or [Datum(model_input=ModelInput.from_ints([0]))]
    steps = int(stage.get("steps", 1))
    for _ in range(steps):
        if budget_seconds is not None and (time.time() - start) > budget_seconds:
            break
        result = client.forward_backward(batch, loss_fn)
        total_loss += result.loss
        client.optim_step(AdamParams(learning_rate=lr))
        total_steps += 1
    ckpt = client.save_weights_for_sampler(name=stage.get("name", f"stage_{total_steps}"))
    return ckpt, {
        "total_steps": total_steps,
        "avg_loss": total_loss / max(1, total_steps),
        "stage": stage.get("name"),
        "loss_fn": loss_fn,
    }


# ── Real path (driven through TrainingClient protocol) ──────────────────

def _run_real_stage(
    workspace: Any,
    stage: dict,
    optimizer: dict | None,
    budget_seconds: float | None,
    *,
    training_client: HFTrainingClient | None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Drive ``TrainingClient.forward_backward("cross_entropy") + optim_step``.

    Dataset is produced by :func:`render_hf_dataset` (same tokenization as
    before); each row becomes a :class:`Datum` with ``attention_mask`` and
    ``labels`` (``-100``-masked prompt) packed into ``loss_fn_inputs``.
    Grad-accumulation, batching and LR warmup are applied in this runner
    (HF Trainer is no longer in the loop).

    DDP dispatch: when ``AE_TRAIN_DDP=1`` (and no caller-provided training
    client), fan out to ``torch.distributed.run`` so all visible GPUs
    participate. Otherwise stay on the verified single-process path.
    """
    import os as _os
    if _os.environ.get("AE_TRAIN_DDP", "0") == "1" and training_client is None:
        return _run_real_stage_ddp(workspace, stage, optimizer, budget_seconds)

    from .data_worker import render_hf_dataset

    cfg = _load_real_stage_config(workspace, stage, optimizer)

    client = training_client or build_hf_client_from_workspace(workspace)

    print(f"[sft] using tokenizer from client ({cfg['model_path']})")
    ds = render_hf_dataset(workspace, client.tokenizer, max_len=cfg["max_seq_len"])
    ds = ds.shuffle(seed=cfg["seed"])
    print(f"[sft] dataset size: {len(ds)}; lr={cfg['lr']}; epochs={cfg['epochs']}")

    per_device_bs = int(cfg["per_device_bs"])
    grad_accum = int(cfg["grad_accum"])
    max_steps = int(cfg["max_steps"]) if cfg["max_steps"] is not None else -1
    warmup_ratio = float(cfg["warmup_ratio"])
    base_lr = float(cfg["lr"])
    epochs = int(cfg["epochs"])
    log_every = max(1, int(cfg["log_every"]))

    rng = random.Random(cfg["seed"])

    def _iter_epoch() -> Iterable[list[int]]:
        order = list(range(len(ds)))
        rng.shuffle(order)
        for chunk_start in range(0, len(order), per_device_bs):
            yield order[chunk_start : chunk_start + per_device_bs]

    total_micro = max(1, (len(ds) // per_device_bs) * epochs)
    total_opt_steps = max(1, total_micro // grad_accum)
    if max_steps > 0:
        total_opt_steps = min(total_opt_steps, max_steps)
    warmup_steps = max(1, int(total_opt_steps * warmup_ratio))

    print(
        f"[sft] total_opt_steps={total_opt_steps}, warmup_steps={warmup_steps}, "
        f"grad_accum={grad_accum}"
    )

    t0 = time.time()
    micro = 0
    opt_step = 0
    accum_loss = 0.0
    losses: list[float] = []

    def _current_lr(step: int) -> float:
        if step < warmup_steps:
            return base_lr * (step + 1) / max(1, warmup_steps)
        # Cosine decay from base_lr → 0 over remaining steps.
        import math

        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    stop = False
    for epoch in range(epochs):
        if stop:
            break
        for batch_idx in _iter_epoch():
            if budget_seconds is not None and (time.time() - t0) > budget_seconds:
                print(f"[sft] budget {budget_seconds}s exceeded — stopping")
                stop = True
                break
            batch = [_row_to_datum(ds[i]) for i in batch_idx]
            result = client.forward_backward(
                batch,
                loss_fn="cross_entropy",
                loss_config={"grad_accum": grad_accum},
            )
            accum_loss += float(result.loss)
            micro += 1
            if micro % grad_accum == 0:
                lr = _current_lr(opt_step)
                client.optim_step(AdamParams(learning_rate=lr))
                opt_step += 1
                mean_loss = accum_loss / grad_accum
                losses.append(mean_loss)
                accum_loss = 0.0
                if opt_step == 1 or opt_step % log_every == 0:
                    elapsed = (time.time() - t0) / 60.0
                    print(
                        f"  step {opt_step}/{total_opt_steps} "
                        f"loss={mean_loss:.4f} lr={lr:.2e} "
                        f"elapsed={elapsed:.1f}min"
                    )
                if max_steps > 0 and opt_step >= max_steps:
                    stop = True
                    break

    wall_seconds = time.time() - t0
    ckpt = client.save_weights_for_sampler(stage.get("name", "sft"))
    print(f"[sft] saved adapter to {ckpt.path} in {wall_seconds:.1f}s")

    # If we own the client, release GPU memory so the subsequent vLLM eval
    # has headroom. If the caller owns it (passed in), they decide when to
    # close — don't tear it down out from under them.
    if training_client is None:
        client.close()

    avg_loss = sum(losses) / max(1, len(losses))
    return (
        CheckpointRef(
            name=stage.get("name", "sft"),
            path=ckpt.path,
            kind="adapter",
            metadata={"lr": base_lr, "rank": cfg["rank"]},
        ),
        {
            "total_steps": opt_step,
            "avg_loss": avg_loss,
            "stage": stage.get("name"),
            "loss_fn": stage.get("loss", "cross_entropy"),
            "lr": base_lr,
            "wall_seconds": wall_seconds,
        },
    )


def _row_to_datum(row: dict[str, Any]) -> Datum:
    """Convert a :func:`render_hf_dataset` row into a ``Datum``."""
    return Datum(
        model_input=ModelInput.from_ints(list(row["input_ids"])),
        loss_fn_inputs={
            "attention_mask": list(row["attention_mask"]),
            "labels": list(row["labels"]),
        },
    )


# ── Config plumbing ─────────────────────────────────────────────────────

def _load_real_stage_config(workspace: Any, stage: dict, optimizer: dict | None) -> dict:
    """Assemble the full config dict the HF Trainer needs."""
    base_cfg = _load_yaml_safely(Path(workspace.root) / "model" / "base.yaml")
    adapter_cfg = _load_yaml_safely(Path(workspace.root) / "model" / "adapter.yaml")
    batching_cfg = _load_yaml_safely(Path(workspace.root) / "train" / "batching.yaml")
    optimizer = optimizer or _load_yaml_safely(Path(workspace.root) / "train" / "optimizer.yaml")

    model_path = base_cfg.get("path")
    if not model_path:
        raise RuntimeError(
            f"model/base.yaml::path is required for real SFT; got {base_cfg}"
        )

    return {
        "model_path": model_path,
        "rank": int(adapter_cfg.get("rank", 16)),
        "alpha": int(adapter_cfg.get("alpha", 32)),
        "dropout": float(adapter_cfg.get("dropout", 0.05)),
        "target_modules": list(
            adapter_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"],
            )
        ),
        "lr": float(optimizer.get("lr", 5e-5)),
        "warmup_ratio": float(optimizer.get("warmup_ratio", 0.03)),
        "epochs": int(stage.get("epochs", 2)),
        "max_steps": stage.get("max_steps"),
        "per_device_bs": int(batching_cfg.get("per_device_bs", 1)),
        "grad_accum": int(batching_cfg.get("grad_accum", 8)),
        "max_seq_len": int(batching_cfg.get("max_seq_len", 2560)),
        "seed": int(stage.get("seed", 42)),
        "log_every": int(batching_cfg.get("log_every", 5)),
    }


def _load_yaml_safely(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:  # pragma: no cover — best-effort config load
        return {}


# ── DDP dispatch (used when AE_TRAIN_DDP=1) ─────────────────────────────

def _run_real_stage_ddp(
    workspace: Any,
    stage: dict,
    optimizer: dict | None,
    budget_seconds: float | None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Dispatch real SFT to a torchrun-based DDP subprocess.

    Falls back to the in-process HFTrainingClient path only if the caller
    explicitly passes a training_client (which it wouldn't under DDP mode).
    """
    from ...backends.tinkerlite.ddp_launcher import run_sft_ddp

    root = Path(workspace.root)
    base_cfg = _load_yaml_safely(root / "model" / "base.yaml")
    adapter_cfg = _load_yaml_safely(root / "model" / "adapter.yaml")
    batching_cfg = _load_yaml_safely(root / "train" / "batching.yaml")
    opt_cfg = optimizer or _load_yaml_safely(root / "train" / "optimizer.yaml")

    # Resolve starting adapter (if any) via the same convention the rest of the
    # backend uses: model/adapter.yaml::seed_adapter_path.
    start_adapter = adapter_cfg.get("seed_adapter_path") or None
    if start_adapter:
        start = Path(start_adapter)
        if not start.is_absolute():
            start = (root / start_adapter).resolve()
        start_adapter = str(start) if start.is_dir() else None

    return run_sft_ddp(
        workspace,
        stage,
        base_cfg=base_cfg,
        adapter_cfg=adapter_cfg,
        optimizer_cfg=opt_cfg,
        batching_cfg=batching_cfg,
        start_adapter_path=start_adapter,
        budget_seconds=budget_seconds,
    )
