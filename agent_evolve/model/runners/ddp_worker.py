"""Multi-rank DDP training worker for SFT and GSPO update.

Spawned as ``torchrun --nproc_per_node=N -m agent_evolve.model.runners.ddp_worker``
by the parent process (single Python process running the MCGS driver). Each rank:

  1. Reads ``--config`` (a JSON with stage kind, paths, hyperparameters).
  2. Sets up DDP via ``torch.distributed.init_process_group("nccl")``.
  3. Loads the base model in bf16 on its assigned GPU.
  4. Attaches (or loads) the LoRA adapter; wraps in ``DistributedDataParallel``.
  5. Runs the stage (SFT or GSPO) on its shard of the dataset.
  6. DDP auto-syncs gradients on every ``.backward()``; optim.step runs per-rank
     with already-synced grads.
  7. Rank 0 saves the adapter + writes ``result.json`` with metrics.
  8. All ranks exit.

This mirrors TRL's ``GRPOTrainer`` design: rewards/advantages are computed
globally then sliced per rank, and DDP handles the gradient sync.

Not a ``TrainingClient`` protocol implementation — the protocol is for the
in-process case. The parent dispatches to this worker via subprocess, waits
for exit, then reads the saved adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any


# ── Env hygiene ─────────────────────────────────────────────────────────

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
# NCCL: match vLLM-style settings so we don't accidentally share a socket.
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Helpers ─────────────────────────────────────────────────────────────

def _print(rank: int, msg: str) -> None:
    """Print with rank prefix; rank 0 only for clean logs."""
    if rank == 0:
        print(f"[ddp-worker rank0] {msg}", flush=True)


def _log_all_ranks(rank: int, msg: str) -> None:
    print(f"[ddp-worker rank{rank}] {msg}", flush=True)


def _build_model_and_optim(cfg: dict, rank: int, world_size: int):
    """Load base + LoRA adapter on this rank's GPU, wrap in DDP, return (model, tokenizer, optimizer).

    Order matters: PEFT adapter load runs BEFORE ``dist.init_process_group``
    so we avoid PEFT's TP-sharding code path (which requires a newer
    transformers than we have). DDP wrap happens AFTER process-group init.
    """
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _print(rank, f"loading tokenizer from {cfg['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _print(rank, "loading base model in bf16")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    # After CUDA_VISIBLE_DEVICES masking in main(), each rank sees exactly
    # one GPU (exposed as cuda:0). Physical mapping is local_rank → cuda:0.
    base = base.to("cuda:0")
    base.config.use_cache = False

    start_adapter = cfg.get("start_adapter_path")
    if start_adapter and Path(start_adapter).is_dir():
        _print(rank, f"loading start adapter from {start_adapter}")
        # Adapter load MUST happen before init_process_group to skip the
        # PEFT TP auto-sharding branch.
        model = PeftModel.from_pretrained(base, start_adapter, is_trainable=True)
    else:
        _print(rank, f"attaching fresh LoRA rank={cfg['lora_rank']}")
        lora_cfg = LoraConfig(
            r=int(cfg["lora_rank"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg["target_modules"]),
        )
        model = get_peft_model(base, lora_cfg)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    if rank == 0:
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    # Now safe to init DDP — adapter weights are already on each rank's GPU.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        _print(rank, f"DDP init done (after PEFT load)")

    # DDP wrap. Only LoRA params are trainable; wrap the full model so DDP
    # sees them through .parameters(). find_unused_parameters=True is needed
    # because LoRA adds params only on some modules (others stay frozen).
    # Each rank sees exactly one GPU (cuda:0 after the CVD mask).
    ddp_model = DDP(
        model,
        device_ids=[0],
        output_device=0,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )

    trainable = [p for p in ddp_model.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(trainable, lr=float(cfg["lr"]), fused=True)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(trainable, lr=float(cfg["lr"]))
    optimizer.zero_grad()

    _print(rank, f"model ready (cuda:0 after CVD mask), "
                 f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M")
    return ddp_model, tokenizer, optimizer


def _save_rank0(ddp_model, tokenizer, outdir: Path) -> None:
    """Rank-0 saves the PEFT adapter (unwrapping DDP first)."""
    outdir.mkdir(parents=True, exist_ok=True)
    inner = ddp_model.module  # unwrap DDP
    inner.save_pretrained(str(outdir))
    tokenizer.save_pretrained(str(outdir))
    if not (outdir / "adapter_config.json").is_file():
        raise RuntimeError(f"save_pretrained did not emit adapter_config.json under {outdir}")


# ── SFT stage ───────────────────────────────────────────────────────────

def _run_sft(cfg: dict, rank: int, world_size: int) -> dict:
    """Cross-entropy SFT across ranks. Each rank takes a DistributedSampler slice."""
    import torch

    ddp_model, tokenizer, optimizer = _build_model_and_optim(cfg, rank, world_size)

    # Lazy import so single-process tests don't pull in datasets when unused.
    sys.path.insert(0, str(Path(cfg["ae_root"])))
    from agent_evolve.model.runners.helpers.dataset import PadToLongest, render_hf_dataset
    from agent_evolve.model.workspace import TrainingWorkspace

    ws = TrainingWorkspace.load(cfg["workspace_root"])
    ds = render_hf_dataset(ws, tokenizer, max_len=int(cfg["max_seq_len"]))
    ds = ds.shuffle(seed=int(cfg["seed"]))
    _print(rank, f"dataset size={len(ds)}; lr={cfg['lr']}; epochs={cfg['epochs']}")

    per_device_bs = int(cfg["per_device_bs"])
    grad_accum = int(cfg["grad_accum"])
    max_steps = int(cfg["max_steps"]) if cfg["max_steps"] is not None else -1
    warmup_ratio = float(cfg["warmup_ratio"])
    base_lr = float(cfg["lr"])
    epochs = int(cfg["epochs"])
    log_every = max(1, int(cfg["log_every"]))

    # Sharded per-rank indices. Standard DDP-sampler pattern.
    def _rank_epoch_indices(epoch: int) -> list[list[int]]:
        rng = random.Random(int(cfg["seed"]) + epoch)
        order = list(range(len(ds)))
        rng.shuffle(order)
        # Trim so every rank gets an equal number of examples.
        trimmed = (len(order) // world_size) * world_size
        order = order[:trimmed]
        shard = order[rank::world_size]  # strided shard
        return [shard[i : i + per_device_bs] for i in range(0, len(shard), per_device_bs)]

    total_micro = max(1, (len(ds) // (per_device_bs * world_size)) * epochs)
    total_opt_steps = max(1, total_micro // grad_accum)
    if max_steps > 0:
        total_opt_steps = min(total_opt_steps, max_steps)
    warmup_steps = max(1, int(total_opt_steps * warmup_ratio))

    _print(rank, f"total_opt_steps={total_opt_steps} warmup={warmup_steps} "
                 f"per_device_bs={per_device_bs} grad_accum={grad_accum} world={world_size}")

    collator = PadToLongest(pad_token_id=tokenizer.pad_token_id)

    def _current_lr(step: int) -> float:
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    device = "cuda:0"  # after CVD mask, each rank sees one GPU
    t0 = time.time()
    micro = 0
    opt_step = 0
    accum_loss = 0.0
    losses: list[float] = []
    stop = False

    for epoch in range(epochs):
        if stop:
            break
        for batch_idx in _rank_epoch_indices(epoch):
            if cfg.get("budget_seconds") and (time.time() - t0) > float(cfg["budget_seconds"]):
                _print(rank, f"budget exceeded, stopping at opt_step={opt_step}")
                stop = True
                break
            batch = collator([{
                "input_ids": list(ds[i]["input_ids"]),
                "attention_mask": list(ds[i]["attention_mask"]),
                "labels": list(ds[i]["labels"]),
            } for i in batch_idx])
            x = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lab = batch["labels"].to(device)

            out = ddp_model(input_ids=x, attention_mask=am, labels=lab)
            loss = out.loss
            (loss / grad_accum).backward()
            accum_loss += float(loss.detach())
            micro += 1

            if micro % grad_accum == 0:
                lr = _current_lr(opt_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()
                opt_step += 1
                mean_loss = accum_loss / grad_accum
                losses.append(mean_loss)
                accum_loss = 0.0
                if opt_step == 1 or opt_step % log_every == 0:
                    _print(rank, f"step {opt_step}/{total_opt_steps} "
                                 f"loss={mean_loss:.4f} lr={lr:.2e} "
                                 f"elapsed={(time.time()-t0)/60:.1f}min")
                if max_steps > 0 and opt_step >= max_steps:
                    stop = True
                    break

    wall_seconds = time.time() - t0
    if rank == 0:
        outdir = Path(cfg["out_adapter_dir"])
        _save_rank0(ddp_model, tokenizer, outdir)
        _print(rank, f"saved adapter to {outdir} in {wall_seconds:.1f}s")

    return {
        "total_steps": opt_step,
        "avg_loss": sum(losses) / max(1, len(losses)),
        "lr": base_lr,
        "wall_seconds": wall_seconds,
        "world_size": world_size,
    }


# ── GSPO stage ──────────────────────────────────────────────────────────

def _run_gspo(cfg: dict, rank: int, world_size: int) -> dict:
    """GSPO update across ranks on pre-computed rollouts.

    Parent passes a JSONL of rollout records (pid, completion_tokens,
    logprobs_old, advantage, prompt_ids). Each rank takes a contiguous
    slice of the shuffled record list. DDP auto-syncs on ``backward``.
    """
    import torch

    ddp_model, tokenizer, optimizer = _build_model_and_optim(cfg, rank, world_size)
    device = "cuda:0"  # after CVD mask, each rank sees one GPU

    # Load rollouts — all ranks read the same file but process disjoint shards.
    rollouts_path = Path(cfg["rollouts_path"])
    records: list[dict] = []
    with open(rollouts_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    # Shuffle deterministically so every rank sees the same ordering, then shard.
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(records)
    # Trim so every rank gets equal work.
    trimmed = (len(records) // world_size) * world_size
    records = records[:trimmed]
    my_records = records[rank::world_size]  # strided shard
    _print(rank, f"world={world_size} trimmed={len(records)} per_rank={len(my_records)}")

    grad_accum = int(cfg["grad_accum"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    eps_low = float(cfg["eps_low"])
    eps_high = float(cfg["eps_high"])
    token_level = bool(cfg["dapo_token_level"])
    max_steps = cfg.get("max_steps")
    log_every = max(1, int(cfg["log_every"]))

    t0 = time.time()
    micro = 0
    opt_steps = 0
    accum_loss = 0.0
    accum_s = 0.0
    accum_clip = 0.0
    stop = False

    for epoch in range(epochs):
        if stop:
            break
        for r in my_records:
            prompt_ids = list(r["prompt_ids"])
            compl_ids = list(r["completion_tokens"])
            lp_old_list = list(r["logprobs_old"])
            advantage = float(r["advantage"])

            prompt_len = len(prompt_ids)
            full_ids = prompt_ids + compl_ids
            x = torch.tensor([full_ids], dtype=torch.long, device=device)
            out = ddp_model(x)
            logits = out.logits
            pred_logits = logits[0, prompt_len - 1 : -1, :]
            compl_t = torch.tensor(compl_ids, dtype=torch.long, device=device)
            if pred_logits.shape[0] != compl_t.shape[0]:
                # Skip malformed rollout rather than crash the whole cycle.
                continue
            lp_new = torch.nn.functional.log_softmax(
                pred_logits.float(), dim=-1
            ).gather(-1, compl_t.unsqueeze(-1)).squeeze(-1)
            lp_old = torch.tensor(lp_old_list, dtype=torch.float32, device=device)

            if token_level:
                log_ratio_tok = lp_new - lp_old
                s_tok = torch.exp(log_ratio_tok)
                s_tok_clipped = torch.clamp(s_tok, min=1.0 - eps_low, max=1.0 + eps_high)
                obj = torch.min(s_tok * advantage, s_tok_clipped * advantage).mean()
                s_scalar = float(s_tok.detach().mean())
                clipped_flag = float((s_tok.detach() != s_tok_clipped.detach()).any())
            else:
                log_ratio = (lp_new - lp_old).mean()
                s = torch.exp(log_ratio)
                s_clipped = torch.clamp(s, min=1.0 - eps_low, max=1.0 + eps_high)
                obj = torch.min(s * advantage, s_clipped * advantage)
                s_scalar = float(s.detach())
                clipped_flag = float(s.detach() != s_clipped.detach())

            loss = -(obj) / grad_accum
            loss.backward()
            accum_loss += float(loss.detach()) * grad_accum
            accum_s += s_scalar
            accum_clip += clipped_flag
            micro += 1

            if micro % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                opt_steps += 1
                if opt_steps == 1 or opt_steps % log_every == 0:
                    _print(rank, f"step {opt_steps} loss={accum_loss/grad_accum:.4f} "
                                 f"mean_s={accum_s/grad_accum:.4f} "
                                 f"clip_frac={accum_clip/grad_accum:.2f} "
                                 f"elapsed={(time.time()-t0)/60:.1f}min")
                accum_loss = accum_s = accum_clip = 0.0
                if max_steps is not None and opt_steps >= int(max_steps):
                    stop = True
                    break

    wall_seconds = time.time() - t0
    if rank == 0:
        outdir = Path(cfg["out_adapter_dir"])
        _save_rank0(ddp_model, tokenizer, outdir)
        _print(rank, f"saved adapter to {outdir}")

    return {
        "total_rollouts": len(records),
        "per_rank_rollouts": len(my_records),
        "opt_steps": opt_steps,
        "wall_seconds": wall_seconds,
        "world_size": world_size,
        "lr": lr,
    }


# ── Entry point ─────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Rank isolation: mask CUDA_VISIBLE_DEVICES to this rank's physical GPU
    # BEFORE any torch import. Without this, transformers/accelerate/NCCL
    # init paths create ~520 MiB CUDA contexts on every *other* GPU (because
    # from_pretrained implicitly touches cuda:0 during module init, and NCCL
    # opens handles to all peers). With the mask, each rank sees exactly one
    # GPU (as cuda:0), so it can't leak contexts onto sibling ranks' GPUs.
    # NCCL still works: torchrun's master_addr/master_port are socket-based,
    # independent of CUDA visibility.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    # If torchrun already set CUDA_VISIBLE_DEVICES (rare), respect it; otherwise
    # narrow to this rank's physical GPU.
    if "CUDA_VISIBLE_DEVICES" not in os.environ or "," in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    import torch
    import torch.distributed as dist

    # After masking, there's exactly one visible GPU per rank — cuda:0. All
    # downstream ``cuda:{rank}``-style calls in this worker must be rewritten
    # to ``cuda:0``. Use a module-level attribute that ``_build_model_and_optim``
    # reads, keeping the rest of the code device-agnostic.
    torch.cuda.set_device(0)

    # IMPORTANT: we do NOT init the process group yet. PEFT's
    # `set_peft_model_state_dict` checks ``dist.is_initialized()`` and, if
    # true, routes through a transformers.integrations.tensor_parallel import
    # that requires a newer transformers version than we have pinned. Loading
    # the adapter first (on each rank, before DDP init) sidesteps that path
    # entirely — each rank loads the full LoRA adapter onto its own GPU, and
    # DDP takes over once we wrap in ``DistributedDataParallel``.
    _print(rank, f"pre-DDP model load: rank={rank} world={world_size} local={local_rank}")

    kind = cfg["kind"]
    if kind == "sft":
        result = _run_sft(cfg, rank, world_size)
    elif kind == "gspo":
        result = _run_gspo(cfg, rank, world_size)
    else:
        raise ValueError(f"Unknown kind={kind!r}")

    # Synchronize before rank 0 writes the result JSON.
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        result_path = Path(cfg["out_result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        _print(rank, f"wrote result to {result_path}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
