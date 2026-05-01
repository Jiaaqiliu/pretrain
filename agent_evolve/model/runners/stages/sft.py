"""SFT training worker.

Three paths, dispatched by ``model/adapter.yaml::type`` →
``ModelAdapter.attach_mode``:

* **smoke** — uses :class:`MockTrainingClient`, no torch. Used by PR8 seed and
  unit tests.
* **real, ``ATTACH_MODE_WRAP``** (e.g. LoRA) — drives an
  :class:`HFTrainingClient` via the ``TrainingClient`` protocol
  (``forward_backward("cross_entropy") + optim_step()``). Matches the
  verified recipe in ``../nemotron-auto-research/scripts/train_sft_lora.py``
  without going through :class:`transformers.Trainer`.
* **real, ``ATTACH_MODE_INPLACE``** (e.g. full-param + DeepSpeed ZeRO-3) —
  bypasses :class:`HFTrainingClient` and drives HF
  :class:`~transformers.Trainer` directly so DeepSpeed can shard the
  optimizer state. The :class:`ModelAdapter` is asked to ``attach()`` (a
  no-op wrap; just gradient checkpointing) and ``save()`` (full
  ``model.safetensors``). The eval stage discovers
  ``CheckpointRef.kind == "full_state"`` and skips ``LoRARequest``.

Adding a new fine-tuning surface is one ``@register_adapter("<kind>")``
file; the dispatch in :func:`_run_real_stage` already routes both
attach modes correctly.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Iterable, Protocol

import yaml

from ....backends.tinkerlite.adapters import (
    ATTACH_MODE_INPLACE,
    resolve_adapter,
)
from ....backends.tinkerlite.base import AdamParams, Datum, ModelInput, TrainingClient
from ....backends.tinkerlite.clients.hf import build_hf_client_from_workspace
from ....backends.tinkerlite.clients.mock import MockTrainingClient
from ...types import CheckpointRef


class SFTTrainingClient(TrainingClient, Protocol):
    """Training client surface needed by the in-process SFT runner."""

    tokenizer: Any

    def close(self) -> None: ...


# ── Public entrypoint ────────────────────────────────────────────────────

def run_sft_stage(
    workspace: Any,
    stage: dict,
    datums: Iterable[Datum] | None = None,
    *,
    optimizer: dict | None = None,
    smoke: bool = True,
    budget_seconds: float | None = None,
    training_client: SFTTrainingClient | None = None,
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
    training_client: SFTTrainingClient | None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Drive ``TrainingClient.forward_backward("cross_entropy") + optim_step``.

    Dataset is produced by :func:`render_hf_dataset` (same tokenization as
    before); each row becomes a :class:`Datum` with ``attention_mask`` and
    ``labels`` (``-100``-masked prompt) packed into ``loss_fn_inputs``.
    Grad-accumulation, batching and LR warmup are applied in this runner
    (HF Trainer is no longer in the loop).

    Adapter dispatch: ``model/adapter.yaml::type`` resolves to a
    :class:`ModelAdapter`. ``attach_mode == ATTACH_MODE_INPLACE`` (e.g.
    full-param) flips this stage onto the HF Trainer path; the LoRA-style
    ``ATTACH_MODE_WRAP`` adapters keep using the step-driven HF client.

    DDP dispatch: when ``AE_TRAIN_DDP=1`` (and no caller-provided training
    client), fan out to ``torch.distributed.run`` so all visible GPUs
    participate. Otherwise stay on the verified single-process path.
    """
    import os as _os

    # Adapter-driven dispatch happens *before* DDP dispatch: a full-param
    # ``ATTACH_MODE_INPLACE`` adapter has its own torch/Trainer launch and
    # does not go through ``ddp_launcher.run_sft_ddp``.
    if _resolve_attach_mode(workspace) == ATTACH_MODE_INPLACE:
        return _run_real_stage_full_param(workspace, stage, optimizer, budget_seconds)

    if _os.environ.get("AE_TRAIN_DDP", "0") == "1" and training_client is None:
        return _run_real_stage_ddp(workspace, stage, optimizer, budget_seconds)

    from ..helpers.dataset import render_hf_dataset

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


# ── Adapter resolution ──────────────────────────────────────────────────

def _resolve_attach_mode(workspace: Any) -> str:
    """Return the registered adapter's ``attach_mode``.

    Default to ``ATTACH_MODE_WRAP`` if the workspace lacks
    ``model/adapter.yaml`` or the file omits ``type`` — that's the legacy
    LoRA path. Unknown ``type`` strings raise (loudly) via
    :func:`resolve_adapter`.
    """
    from ....backends.tinkerlite.adapters import ATTACH_MODE_WRAP

    adapter_cfg = _load_yaml_safely(Path(workspace.root) / "model" / "adapter.yaml")
    kind = adapter_cfg.get("type")
    if not kind:
        return ATTACH_MODE_WRAP
    return resolve_adapter(kind).attach_mode


# ── Real path, ATTACH_MODE_INPLACE (HF Trainer + DeepSpeed) ─────────────

def _run_real_stage_full_param(
    workspace: Any,
    stage: dict,
    optimizer: dict | None,
    budget_seconds: float | None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Full-parameter SFT via HF :class:`Trainer`.

    Uses the ``ModelAdapter`` registered for the workspace (typically
    :class:`FullDeepspeedAdapter`) for ``attach`` (no-op wrap; flip
    ``use_cache`` + gradient checkpointing) and ``save`` (write the full
    ``model.safetensors`` + ``config.json`` + tokenizer). The Trainer
    runs DeepSpeed ZeRO-3 when ``train/deepspeed.json`` is present in the
    workspace, otherwise it runs single-process bf16.

    ``budget_seconds`` is currently advisory: HF Trainer doesn't expose a
    wall-clock cutoff cleanly, and DeepSpeed runs are wall-bounded by
    their k8s job timeout in practice. We log it and let
    ``max_steps`` / ``epochs`` drive termination.
    """
    import gc
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from ..helpers.dataset import PadToLongest, render_hf_dataset

    cfg = _load_full_param_config(workspace, stage, optimizer)
    adapter_cfg = _load_yaml_safely(Path(workspace.root) / "model" / "adapter.yaml")
    adapter = resolve_adapter(str(adapter_cfg.get("type", "full_deepspeed_customized")))

    print(f"[sft-full] loading tokenizer from {cfg['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = render_hf_dataset(workspace, tokenizer, max_len=cfg["max_seq_len"])
    ds = ds.shuffle(seed=cfg["seed"])
    print(f"[sft-full] dataset size: {len(ds)}; lr={cfg['lr']}; epochs={cfg['epochs']}")

    print("[sft-full] loading base model in bf16 (full-param, no LoRA)")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = adapter.attach(base_model, adapter_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[sft-full] params: {total_params:,} total, {trainable_params:,} trainable "
        f"({100.0 * trainable_params / max(1, total_params):.1f}%)"
    )

    outdir = Path(workspace.root) / "checkpoints" / "full_model" / stage.get("name", "sft_full")
    outdir.mkdir(parents=True, exist_ok=True)

    max_steps = int(cfg["max_steps"]) if cfg["max_steps"] is not None else -1

    targs = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=cfg["epochs"] if max_steps <= 0 else 1,
        max_steps=max_steps,
        per_device_train_batch_size=cfg["per_device_bs"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=cfg["log_every"],
        save_strategy="no",
        bf16=True,
        seed=cfg["seed"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        deepspeed=cfg.get("deepspeed_config"),
    )

    collator = PadToLongest(pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

    if budget_seconds is not None:
        print(f"[sft-full] budget_seconds={budget_seconds} (advisory; not enforced)")

    t0 = time.time()
    train_output = trainer.train()
    wall_seconds = time.time() - t0

    # Trainer.save_model writes config.json + weights; the adapter then
    # adds the tokenizer (idempotent re-save_pretrained is harmless).
    trainer.save_model(str(outdir))
    ckpt = adapter.save(model, tokenizer, outdir)
    print(f"[sft-full] saved full model to {ckpt.path} in {wall_seconds:.1f}s")

    try:
        del trainer
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass

    return (
        ckpt,
        {
            "total_steps": int(getattr(train_output, "global_step", 0)),
            "avg_loss": float(getattr(train_output, "training_loss", 0.0) or 0.0),
            "stage": stage.get("name"),
            "loss_fn": stage.get("loss", "cross_entropy"),
            "lr": cfg["lr"],
            "wall_seconds": wall_seconds,
            "adapter_type": adapter.kind,
        },
    )


def _load_full_param_config(workspace: Any, stage: dict, optimizer: dict | None) -> dict:
    """Assemble the config the full-param Trainer path needs.

    Mirrors :func:`_load_real_stage_config` in shape but reads no LoRA
    knobs (rank, alpha, target_modules) — those are meaningless when no
    adapter is being attached. Picks up an optional DeepSpeed config
    from ``train/deepspeed.json`` if present.
    """
    root = Path(workspace.root)
    base_cfg = _load_yaml_safely(root / "model" / "base.yaml")
    batching_cfg = _load_yaml_safely(root / "train" / "batching.yaml")
    optimizer = optimizer or _load_yaml_safely(root / "train" / "optimizer.yaml")

    model_path = base_cfg.get("path")
    if not model_path:
        raise RuntimeError(
            f"model/base.yaml::path is required for real SFT; got {base_cfg}"
        )

    deepspeed_path = root / "train" / "deepspeed.json"
    deepspeed_config = str(deepspeed_path) if deepspeed_path.is_file() else None

    return {
        "model_path": model_path,
        "lr": float(optimizer.get("lr", 1e-5)),
        "warmup_ratio": float(optimizer.get("warmup_ratio", 0.05)),
        "epochs": int(stage.get("epochs", 5)),
        "max_steps": stage.get("max_steps"),
        "per_device_bs": int(batching_cfg.get("per_device_bs", 1)),
        "grad_accum": int(batching_cfg.get("grad_accum", 16)),
        "max_seq_len": int(batching_cfg.get("max_seq_len", 8192)),
        "seed": int(stage.get("seed", 42)),
        "log_every": int(batching_cfg.get("log_every", 5)),
        "deepspeed_config": deepspeed_config,
    }


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
    from ....backends.tinkerlite.single_node.ddp_launcher import run_sft_ddp

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


# ── StageRegistry adapter ────────────────────────────────────────────────
#
# Thin unpack of ``StageContext`` → ``run_sft_stage(...)``. See
# ``training/stage_registry.py`` and ``INTEGRATION.md`` §2.

from ...stage_registry import StageContext, StageResult, register_stage  # noqa: E402


@register_stage("sft")
def _sft_stage_adapter(ctx: StageContext) -> StageResult:
    import os as _os
    from ..helpers.dataset import render_datums

    # Smoke path feeds mock Datums; real path loads its tokenized dataset
    # inside run_sft_stage via render_hf_dataset. DDP dispatch defers
    # training-client construction to the torchrun subprocess.
    datums = list(render_datums(ctx.workspace, smoke=ctx.smoke)) if ctx.smoke else None
    use_ddp = (not ctx.smoke) and _os.environ.get("AE_TRAIN_DDP", "0") == "1"
    client = None
    if ctx.training_client_fn is not None and not (ctx.smoke or use_ddp):
        client = ctx.training_client_fn()

    ckpt, metrics = run_sft_stage(
        ctx.workspace,
        ctx.stage,
        datums,
        optimizer=ctx.optimizer,
        smoke=ctx.smoke,
        budget_seconds=ctx.budget_seconds,
        training_client=client,
    )
    return StageResult(checkpoint=ckpt, metrics=metrics)
