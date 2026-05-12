"""Multi-GPU DDP smoke test — Unsloth + Nemotron-H + huikang recipe, 2 steps only.

Launch with:
    accelerate launch --num_processes 8 --mixed_precision bf16 /tmp/unsloth_ddp_smoke.py

Pass criteria:
  - All 8 processes load the model
  - LoRA wraps successfully
  - 2 optimizer steps complete (training loss produced)
  - No NCCL / desync / OOM errors

Uses 16 training rows (2 per rank × 8 ranks) so the smoke is fast.
"""
from __future__ import annotations
# IMPORTANT: unsloth must import first for its patches to be applied before
# trl/transformers/peft are imported (warning emitted otherwise).
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import os, json, time, sys, random, math

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HFDataset

# ── Config (huikang published recipe, truncated for smoke) ─────────────
MODEL_PATH   = "/fsx/models/Nemotron-3-Nano-30B-A3B-unsloth"
MAX_SEQ_LEN  = 4096   # huikang uses 8192; shorter here = faster smoke
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj",
    "in_proj", "out_proj",
    "lm_head",
]
LR           = 2e-4
PER_DEV_BS   = 1
GRAD_ACCUM   = 1       # smoke: minimal grad_accum
NUM_STEPS    = 2       # smoke: two optimizer steps only
SEED         = 42

TRAIN_JSONL  = "/fsx/zzsamshi/a-evolve/runs/nemo-mas-teams-v2/cycles/0002/.fork_target/nodes/workspace/workspace/data/final/train.jsonl"


def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes
    device = accelerator.device

    def log(msg):
        accelerator.print(f"[rank {rank}/{world}] {msg}", flush=True)

    set_seed(SEED)
    # Pin this rank to its local GPU BEFORE importing/loading any CUDA code.
    # Without this, every rank sees all 8 GPUs and FastLanguageModel's device_map="auto"
    # puts the whole model on cuda:0, OOMing it with 8 × 17 GB.
    import torch.cuda as _cuda
    _cuda.set_device(accelerator.local_process_index)
    log(f"pinned to cuda:{accelerator.local_process_index}")

    t0 = time.time()
    log("loading model via Unsloth FastLanguageModel.from_pretrained ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map={"": accelerator.local_process_index},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"model loaded in {time.time()-t0:.1f}s, device={model.device}")

    log("wrapping LoRA ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        # Critical: cannot use "unsloth" gradient checkpointing with DDP.
        use_gradient_checkpointing=True,
        random_state=SEED,
    )
    FastLanguageModel.for_training(model)

    # Nemotron-H Mamba fast path patch (per rank)
    _nemo = None
    for nm, m in sys.modules.items():
        if "modeling_nemotron_h" in nm and hasattr(m, "is_fast_path_available"):
            _nemo = m; break
    if _nemo is not None:
        _nemo.is_fast_path_available = True
        log(f"Mamba fast path: ON")
    else:
        log("WARN: modeling_nemotron_h not patched")

    # fp32 LoRA cast
    n_fp32 = 0
    for name, p in model.named_parameters():
        if ".lora_" in name:
            p.data = p.data.to(torch.float32)
            n_fp32 += 1
    log(f"cast {n_fp32} LoRA tensors to fp32")

    # Load a small training set — just enough for 2 steps
    N_ROWS = max(16, world * PER_DEV_BS * GRAD_ACCUM * NUM_STEPS + 4)
    rows = []
    with open(TRAIN_JSONL) as f:
        for i, line in enumerate(f):
            if i >= N_ROWS: break
            r = json.loads(line)
            rows.append(r)
    log(f"loaded {len(rows)} training rows")

    # Build dataset with prompt/completion split so trl can build a
    # completion-only loss mask itself (matches huikang/dgxchen).
    records = []
    for r in rows:
        records.append({"prompt": r["prompt_rendered"], "completion": r["completion"]})
    dataset = HFDataset.from_list(records)

    args = SFTConfig(
        output_dir="/tmp/unsloth_ddp_out",
        num_train_epochs=1,
        max_steps=NUM_STEPS,
        per_device_train_batch_size=PER_DEV_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_length=MAX_SEQ_LEN,
        adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
        weight_decay=0.0, max_grad_norm=1e9,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=SEED,
        report_to="none",
        packing=False,
        # MoE + lm_head LoRA means some params don't see a grad every step.
        ddp_find_unused_parameters=True,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    log("starting trainer.train() for 2 steps ...")
    t1 = time.time()
    result = trainer.train()
    log(f"trainer.train() completed in {time.time()-t1:.1f}s, loss={getattr(result,'metrics',{}).get('train_loss','?')}")

    accelerator.wait_for_everyone()
    log("PASS — 8-GPU DDP Unsloth smoke complete")


if __name__ == "__main__":
    main()
