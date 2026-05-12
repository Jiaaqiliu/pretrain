"""Unsloth-backed SFT stage.

Alternative to the platform's ``sft`` stage, matching the recipe top public
Kaggle notebooks use to reach ~0.85 LB on the NVIDIA Nemotron Reasoning
challenge (dgxchen, konbu17, huikang). Drives ``trl.SFTTrainer`` on top of
``unsloth.FastLanguageModel`` so LoRA fine-tuning of Nemotron-3-Nano-30B-A3B
fits on a single GPU at 8k context. Expects chat rows (``{"messages": [...]}``).

Pinned runtime: unsloth, trl, peft, transformers, datasets, accelerate,
bitsandbytes, mamba_ssm==2.3.1, causal_conv1d==1.6.1.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from ...stage_registry import StageContext, StageResult, register_stage
from ...types import CheckpointRef


# huikang 0.85-LB defaults, applied when ``use_published_recipe: true``.
# Matches end-to-end-finetuning-for-lb-0-85 (huikang). The lm_head entry is
# NOT a no-op — our post-wrap code manually LoRA-injects lm_head because
# Unsloth drops it from MoE targets even when requested.
_PUBLISHED_RECIPE = {
    "lr": 2e-4,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "warmup_ratio": 0.0,
    "per_device_bs": 1,
    "grad_accum": 32,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "max_grad_norm": 1e9,
    "rank": 32,
    "alpha": 32,
    "dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj", "up_proj", "down_proj",
        "lm_head",
    ],
    "max_seq_len": 8192,
    "epochs": 1,
    "prompt_suffix": (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    ),
}

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def run_sft_unsloth_stage(
    workspace: Any,
    stage: dict,
    *,
    optimizer: dict | None = None,
    budget_seconds: float | None = None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    cfg = _load_stage_config(workspace, stage, optimizer)
    return _run_real_unsloth_stage(workspace, stage, cfg, budget_seconds)


def _run_real_unsloth_stage(
    workspace: Any,
    stage: dict,
    cfg: dict,
    budget_seconds: float | None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    # Deferred imports so the module stays importable on CPU-only nodes.
    import torch
    from datasets import Dataset as HFDataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from ..helpers.dataset import load_chat_rows

    t0 = time.time()
    workspace_root = Path(workspace.root)
    seed = int(cfg["seed"])

    print(f"[sft_unsloth] loading base model from {cfg['model_path']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_path"],
        max_seq_length=cfg["max_seq_len"],
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"[sft_unsloth] wrapping LoRA: r={cfg['rank']}, alpha={cfg['alpha']}, "
        f"dropout={cfg['dropout']}, targets={cfg['target_modules']}"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=list(cfg["target_modules"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    FastLanguageModel.for_training(model)

    # ── Nemotron-H specific patches (huikang 0.85-LB recipe) ─────────────
    # 1) Force Mamba CUDA fast path on. Unsloth's Nemotron_H patching flips
    #    `is_fast_path_available` off which falls back to a ~100× slower
    #    pure-Python SSM loop.
    import sys as _sys
    _nemo_mod = None
    for _name, _m in _sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            _nemo_mod = _m
            break
    if _nemo_mod is not None:
        prev = getattr(_nemo_mod, "is_fast_path_available")
        _nemo_mod.is_fast_path_available = True
        print(f"[sft_unsloth] patched Mamba fast path: {prev} → True")
    else:
        print("[sft_unsloth] WARNING: modeling_nemotron_h module not found; "
              "Mamba fast path not patched — training will be slow")

    # 2) Manually add lm_head LoRA. Unsloth drops lm_head from MoE models'
    #    target_modules even when explicitly requested.
    if "lm_head" in cfg["target_modules"]:
        from peft import LoraConfig
        from peft.tuners.lora import Linear as LoraLinear
        _causal = model
        while hasattr(_causal, "model"):
            _causal = _causal.model
        _lm = _causal.lm_head
        if not isinstance(_lm, LoraLinear):
            _cfg = LoraConfig(
                r=cfg["rank"], lora_alpha=cfg["alpha"],
                lora_dropout=cfg["dropout"],
            )
            model.base_model._create_and_replace(
                _cfg, "default",
                target=_lm, target_name="lm_head", parent=_causal,
            )
            print("[sft_unsloth] manually added LoRA to lm_head")
        else:
            print("[sft_unsloth] lm_head already wrapped in LoRA")

    # 3) Dtype discipline — LoRA params to fp32, base bf16 (MoE router stays
    #    fp32 by construction per NemotronH._keep_in_fp32_modules_strict).
    for _name, _p in model.named_parameters():
        if ".lora_" in _name:
            _p.data = _p.data.to(torch.float32)

    model.print_trainable_parameters()

    rows = load_chat_rows(workspace, split="train")
    if not rows:
        raise ValueError(
            f"[sft_unsloth] no training rows under "
            f"{workspace_root / 'data' / 'sources.yaml'}"
        )
    records, record_types = _build_sft_records(rows, cfg["prompt_suffix"], tokenizer)
    if not records:
        raise ValueError(
            "[sft_unsloth] every row dropped while building SFT records "
            "(each row needs 'messages' with user + assistant turns and a "
            "\\boxed{...} answer)"
        )
    dataset = HFDataset.from_list(records)
    print(f"[sft_unsloth] SFT records: {len(records)} / raw rows: {len(rows)}")

    effective_batch = int(cfg["per_device_bs"]) * int(cfg["grad_accum"])
    training_args = SFTConfig(
        output_dir=str(workspace_root / "checkpoints" / "sft_unsloth_tmp"),
        num_train_epochs=int(cfg["epochs"]),
        per_device_train_batch_size=int(cfg["per_device_bs"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        learning_rate=float(cfg["lr"]),
        lr_scheduler_type=str(cfg["lr_scheduler_type"]),
        warmup_steps=int(cfg["warmup_steps"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        max_length=int(cfg["max_seq_len"]),
        adam_beta1=float(cfg["adam_beta1"]),
        adam_beta2=float(cfg["adam_beta2"]),
        adam_epsilon=float(cfg["adam_eps"]),
        weight_decay=float(cfg["weight_decay"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        logging_steps=int(cfg["log_every"]),
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=seed,
        report_to="none",
        packing=False,
        # MoE experts + lm_head LoRA mean not every param sees a grad every
        # step; DDP needs this to avoid "unused parameters" reduction errors.
        ddp_find_unused_parameters=True,
        completion_only_loss=True,
    )

    stratified_by = cfg.get("stratified_by")
    if stratified_by:
        if stratified_by != "type":
            raise ValueError(
                f"[sft_unsloth] unsupported stratified_by={stratified_by!r}; "
                "only 'type' is implemented"
            )
        missing = sum(1 for t in record_types if t is None)
        if missing:
            raise ValueError(
                f"[sft_unsloth] stratified_by=type requested but {missing} rows "
                "are missing a 'type' field"
            )
        order = _build_stratified_index_order(record_types, effective_batch, seed)
        trainer = _make_stratified_trainer(
            model, training_args, dataset, tokenizer, order
        )
        print(
            f"[sft_unsloth] stratified batching on 'type'; "
            f"effective_batch={effective_batch}, counts={dict(_counts(record_types))}"
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        print(f"[sft_unsloth] random shuffling; effective_batch={effective_batch}")

    print(
        f"[sft_unsloth] training: epochs={cfg['epochs']}, lr={cfg['lr']}, "
        f"scheduler={cfg['lr_scheduler_type']}, max_seq_len={cfg['max_seq_len']}"
    )
    train_start = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_start
    print(f"[sft_unsloth] done in {train_elapsed/60.0:.1f} min")

    adapter_dir = (
        workspace_root / "checkpoints" / "adapters" / stage.get("name", "sft_unsloth")
    )
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _rewrite_adapter_config_for_kaggle(
        adapter_dir, base_model_name_or_path=cfg["base_model_name"]
    )
    print(f"[sft_unsloth] adapter saved to {adapter_dir}")

    wall_seconds = time.time() - t0
    if budget_seconds is not None and wall_seconds > budget_seconds:
        print(
            f"[sft_unsloth] wall={wall_seconds:.0f}s exceeded "
            f"budget={budget_seconds:.0f}s (adapter saved)"
        )

    metrics_log = getattr(train_result, "metrics", {}) or {}
    final_loss = float(
        metrics_log.get("train_loss")
        or metrics_log.get("train_loss_mean")
        or _last_log_loss(trainer)
        or 0.0
    )
    total_steps = _safe_int(getattr(trainer.state, "global_step", 0))

    ckpt = CheckpointRef(
        name=stage.get("name", "sft_unsloth"),
        path=str(adapter_dir),
        kind="adapter",
        metadata={
            "lr": float(cfg["lr"]),
            "rank": int(cfg["rank"]),
            "alpha": int(cfg["alpha"]),
            "target_modules": list(cfg["target_modules"]),
            "base_model": cfg["base_model_name"],
            "stratified": bool(stratified_by),
            "effective_batch": effective_batch,
        },
    )
    stats = {
        "stage": stage.get("name"),
        "num_steps": total_steps,
        "final_loss": final_loss,
        "elapsed_sec": wall_seconds,
        "train_elapsed_sec": train_elapsed,
        "stratified": bool(stratified_by),
        "lora_rank": int(cfg["rank"]),
        "effective_batch": effective_batch,
        "lr": float(cfg["lr"]),
        "lr_scheduler": str(cfg["lr_scheduler_type"]),
    }
    return ckpt, stats


def _build_sft_records(
    rows: list[dict], prompt_suffix: str, tokenizer
) -> tuple[list[dict], list[str | None]]:
    """Turn ``messages`` rows into trl 0.24 prompt/completion records.

    Injects the competition ``\\boxed{}`` suffix into the user turn and
    rewrites the assistant turn to ``<cot_without_boxed>\\n</think>\\n\\boxed{ans}``.
    Renders the chat template once here so we can emit a clean
    ``{"prompt": ..., "completion": ...}`` split — trl then tokenizes both
    sides and masks the prompt from the loss (completion-only SFT).
    Rows without a ``\\boxed{...}`` final answer are dropped.
    """
    records: list[dict] = []
    record_types: list[str | None] = []
    suffix_stripped = prompt_suffix.strip()

    for row in rows:
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        user = next((m for m in msgs if m.get("role") == "user"), None)
        assistant = next((m for m in msgs if m.get("role") == "assistant"), None)
        if user is None or assistant is None:
            continue
        user_content = str(user.get("content", "")).strip()
        assistant_content = str(assistant.get("content", "")).strip()
        if not user_content or not assistant_content:
            continue
        match = _BOXED_RE.search(assistant_content)
        if not match:
            continue
        answer = match.group(1).strip()
        cot_cleaned = _BOXED_RE.sub("", assistant_content).rstrip()
        if len(cot_cleaned) < 5:
            continue
        rebuilt_assistant = f"{cot_cleaned}\n</think>\n\\boxed{{{answer}}}"
        rebuilt_user = (
            user_content
            if user_content.endswith(suffix_stripped)
            else user_content + prompt_suffix
        )
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": rebuilt_user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": rebuilt_user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        completion_text = rebuilt_assistant + (tokenizer.eos_token or "")
        records.append({"prompt": prompt_text, "completion": completion_text})
        record_types.append(row.get("type"))
    return records, record_types


def _build_stratified_index_order(
    labels: list[str | None], batch_size: int, seed: int
) -> list[int]:
    by_label: dict[Any, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)
    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)
    assigned = 0
    for label in sorted(by_label.keys(), key=lambda x: (x is None, x)):
        for idx in by_label[label]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1
    order = [idx for batch in batches for idx in batch]
    if len(order) != len(labels):
        raise ValueError(
            f"[sft_unsloth] stratified size mismatch: {len(order)} vs {len(labels)}"
        )
    return order


def _make_stratified_trainer(
    model, training_args, dataset, tokenizer, formatting_func, order: list[int]
):
    from torch.utils.data import DataLoader
    from trl import SFTTrainer

    class _Sampler:
        def __init__(self, ord_): self._ord = list(ord_)
        def __iter__(self): return iter(self._ord)
        def __len__(self): return len(self._ord)

    class _StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *args, _order, **kwargs):
            super().__init__(*args, **kwargs)
            self._order = _order

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("[sft_unsloth] trainer needs train_dataset")
            if len(self._order) != len(self.train_dataset):
                raise ValueError(
                    "[sft_unsloth] order length mismatches dataset length"
                )
            kw = {
                "batch_size": self.args.per_device_train_batch_size,
                "sampler": _Sampler(self._order),
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "drop_last": self.args.dataloader_drop_last,
            }
            if self.args.dataloader_num_workers > 0:
                kw["prefetch_factor"] = self.args.dataloader_prefetch_factor
            return DataLoader(self.train_dataset, **kw)

    return _StratifiedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        _order=order,
    )


def _rewrite_adapter_config_for_kaggle(
    adapter_dir: Path, *, base_model_name_or_path: str
) -> None:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"[sft_unsloth] expected {cfg_path} after save_pretrained"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = base_model_name_or_path
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def _load_stage_config(
    workspace: Any, stage: dict, optimizer: dict | None
) -> dict:
    ws_root = Path(workspace.root)
    base_cfg = _load_yaml_safely(ws_root / "model" / "base.yaml")
    adapter_cfg = _load_yaml_safely(ws_root / "model" / "adapter.yaml")
    batching_cfg = _load_yaml_safely(ws_root / "train" / "batching.yaml")
    optimizer_cfg = optimizer or _load_yaml_safely(ws_root / "train" / "optimizer.yaml")

    model_path = base_cfg.get("path") or base_cfg.get("name")
    if not model_path:
        raise RuntimeError(
            f"[sft_unsloth] model/base.yaml must set 'path' or 'name'; got {base_cfg}"
        )
    base_model_name = (
        base_cfg.get("name") or "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )

    use_pub = bool(stage.get("use_published_recipe", False))
    optimizer_cfg = optimizer_cfg or {}
    betas = optimizer_cfg.get("betas") or [None, None]

    def pick(key: str, ws_value: Any, default: Any = None):
        if key in stage:
            return stage[key]
        if use_pub and key in _PUBLISHED_RECIPE:
            return _PUBLISHED_RECIPE[key]
        if ws_value is not None:
            return ws_value
        if key in _PUBLISHED_RECIPE:
            return _PUBLISHED_RECIPE[key]
        return default

    return {
        "model_path": model_path,
        "base_model_name": base_model_name,
        "rank": int(pick("rank", adapter_cfg.get("rank"))),
        "alpha": int(pick("alpha", adapter_cfg.get("alpha"))),
        "dropout": float(pick("dropout", adapter_cfg.get("dropout"))),
        "target_modules": list(pick("target_modules", adapter_cfg.get("target_modules"))),
        "lr": float(pick("lr", optimizer_cfg.get("lr"))),
        "lr_scheduler_type": str(pick("lr_scheduler_type", optimizer_cfg.get("lr_scheduler_type"))),
        "warmup_steps": int(pick("warmup_steps", optimizer_cfg.get("warmup_steps"), 0)),
        "warmup_ratio": float(pick("warmup_ratio", optimizer_cfg.get("warmup_ratio"), 0.0)),
        "weight_decay": float(pick("weight_decay", optimizer_cfg.get("weight_decay"))),
        "adam_beta1": float(pick("adam_beta1", betas[0])),
        "adam_beta2": float(pick("adam_beta2", betas[1])),
        "adam_eps": float(pick("adam_eps", optimizer_cfg.get("eps"))),
        "max_grad_norm": float(pick("max_grad_norm", optimizer_cfg.get("max_grad_norm"))),
        "per_device_bs": int(pick("per_device_bs", batching_cfg.get("per_device_bs"))),
        "grad_accum": int(pick("grad_accum", batching_cfg.get("grad_accum"))),
        "max_seq_len": int(pick("max_seq_len", batching_cfg.get("max_seq_len"))),
        "epochs": int(pick("epochs", None)),
        "log_every": int(pick("log_every", batching_cfg.get("log_every"), 10)),
        "seed": int(stage.get("seed", 42)),
        "stratified_by": stage.get("stratified_by"),
        "prompt_suffix": str(stage.get("prompt_suffix", _PUBLISHED_RECIPE["prompt_suffix"])),
    }


def _load_yaml_safely(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _counts(items: Iterable[Any]) -> dict[Any, int]:
    out: dict[Any, int] = defaultdict(int)
    for x in items:
        out[x] += 1
    return dict(sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0])))


def _last_log_loss(trainer) -> float | None:
    for entry in reversed(getattr(trainer.state, "log_history", []) or []):
        if "loss" in entry:
            return float(entry["loss"])
    return None


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


@register_stage("sft_unsloth")
def _sft_unsloth_stage_adapter(ctx: StageContext) -> StageResult:
    if ctx.smoke:
        raise RuntimeError(
            "[sft_unsloth] smoke mode unsupported; needs GPU + unsloth + mamba_ssm. "
            "Use stage type 'sft' for smoke runs."
        )
    ckpt, metrics = run_sft_unsloth_stage(
        ctx.workspace,
        ctx.stage,
        optimizer=ctx.optimizer,
        budget_seconds=ctx.budget_seconds,
    )
    return StageResult(checkpoint=ckpt, metrics=metrics)
