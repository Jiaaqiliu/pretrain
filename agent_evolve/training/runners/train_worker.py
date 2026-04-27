"""SFT training worker.

Two paths:

* **smoke** — uses :class:`MockTrainingClient`, no torch. Used by PR8 seed and
  unit tests.
* **real** — builds an HF/PEFT pipeline: load base model in bf16 with
  ``trust_remote_code=True``, wrap with LoRA, run HF ``Trainer`` over the
  tokenized dataset, save adapter. Matches the verified recipe in
  ``../nemotron-auto-research/scripts/train_sft_lora.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from ...backends.tinkerlite.base import AdamParams, Datum, ModelInput
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
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Run one SFT stage. Returns the checkpoint plus a metrics dict."""
    if smoke:
        return _run_smoke_stage(workspace, stage, datums or [], optimizer, budget_seconds)
    return _run_real_stage(workspace, stage, optimizer, budget_seconds)


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


# ── Real path (HF + PEFT LoRA) ───────────────────────────────────────────

def _run_real_stage(
    workspace: Any,
    stage: dict,
    optimizer: dict | None,
    budget_seconds: float | None,  # noqa: ARG001 — HF Trainer owns its own budget
) -> tuple[CheckpointRef, dict[str, Any]]:
    # Deferred imports so smoke tests don't pay a torch/peft import.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from .data_worker import PadToLongest, render_hf_dataset

    cfg = _load_real_stage_config(workspace, stage, optimizer)

    # ── Tokenizer + dataset ─────────────────────────────────────────
    print(f"[sft] loading tokenizer from {cfg['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = render_hf_dataset(workspace, tokenizer, max_len=cfg["max_seq_len"])
    ds = ds.shuffle(seed=cfg["seed"])
    print(f"[sft] dataset size: {len(ds)}; lr={cfg['lr']}; epochs={cfg['epochs']}")

    # ── Model ───────────────────────────────────────────────────────
    print("[sft] loading base model in bf16")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # mamba layers unaffected; attn prefers eager here
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    try:
        model.print_trainable_parameters()
    except Exception:  # pragma: no cover — print-only helper
        pass

    # ── TrainingArguments ───────────────────────────────────────────
    outdir = Path(workspace.root) / "checkpoints" / "adapters" / stage["name"]
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
        save_strategy="no",  # we call save_model() manually at the end
        bf16=True,
        seed=cfg["seed"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    collator = PadToLongest(pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

    t0 = time.time()
    train_output = trainer.train()
    wall_seconds = time.time() - t0

    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))
    print(f"[sft] saved adapter to {outdir} in {wall_seconds:.1f}s")

    # Clean up GPU memory so the subsequent vLLM eval has headroom.
    # HF Trainer holds references to the model, optimizer, scheduler, and
    # cached activation buffers; dropping them one at a time + collecting +
    # flushing the allocator cache is what frees the ~60 GB back to the pool.
    import gc

    try:
        del trainer
    except Exception:  # pragma: no cover
        pass
    try:
        del model
    except Exception:  # pragma: no cover
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:  # pragma: no cover
        pass

    return (
        CheckpointRef(
            name=stage.get("name", "sft"),
            path=str(outdir),
            kind="adapter",
            metadata={"lr": cfg["lr"], "rank": cfg["rank"]},
        ),
        {
            "total_steps": int(getattr(train_output, "global_step", 0)),
            "avg_loss": float(getattr(train_output, "training_loss", 0.0) or 0.0),
            "stage": stage.get("name"),
            "loss_fn": stage.get("loss", "cross_entropy"),
            "lr": cfg["lr"],
            "wall_seconds": wall_seconds,
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
