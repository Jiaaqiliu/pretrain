"""W4 SFT on 1 GPU via Unsloth — huikang 0.85-LB recipe, clean rebuild.

Launch:
    CUDA_VISIBLE_DEVICES=7 python /tmp/unsloth_w4_1gpu.py

No Accelerate / no DDP / no Trainer-method monkey-patching. This is the
path we already proved works on the host (smoke2 passed). Minimal drift
from huikang's literal Kaggle notebook.
"""
from __future__ import annotations
import unsloth  # noqa: F401 — must come before trl/transformers
from unsloth import FastLanguageModel

import json, os, sys, time
import torch
from datasets import Dataset as HFDataset
from trl import SFTConfig, SFTTrainer

# ── Config ───────────────────────────────────────────────────────────
MODEL_PATH  = "/fsx/models/Nemotron-3-Nano-30B-A3B-unsloth"
TRAIN_JSONL = "/fsx/zzsamshi/a-evolve/runs/nemo-mas-teams-v2/cycles/0002/.fork_target/nodes/workspace/workspace/data/final/train.jsonl"
OUTPUT_DIR  = "/fsx/zzsamshi/a-evolve/runs/nemo-mas-teams-v2/cycles/0002/.fork_target/nodes/workspace/workspace/checkpoints/adapters/sft_w4_unsloth_1gpu"
MAX_SEQ_LEN = 8192
LORA_RANK   = 32
LORA_ALPHA  = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj",
    "in_proj", "out_proj",
    "lm_head",
]
LR          = 2e-4
PER_DEV_BS  = 1
GRAD_ACCUM  = 32          # global batch = 1 × 32 = 32 (huikang literal)
MAX_STEPS   = 0           # 0 = honor epochs; 1 epoch on 14718 rows ≈ 460 steps
EPOCHS      = 1
SEED        = 42
SAVE_EVERY  = 50
WANDB_PROJECT = "nemo-mas-w"
WANDB_RUN_NAME = "unsloth-w4-1gpu"

print(f"[unsloth-w4] loading {MODEL_PATH}", flush=True)
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False, load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=False,
    attn_implementation="eager",
    dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[unsloth-w4] model loaded in {time.time()-t0:.1f}s", flush=True)

print("[unsloth-w4] wrapping LoRA (huikang recipe)", flush=True)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
FastLanguageModel.for_training(model)

# huikang patch 1: Mamba fast path
_nemo = None
for nm, m in sys.modules.items():
    if "modeling_nemotron_h" in nm and hasattr(m, "is_fast_path_available"):
        _nemo = m; break
assert _nemo is not None, "modeling_nemotron_h not loaded"
_nemo.is_fast_path_available = True
print("[unsloth-w4] Mamba fast path: ON", flush=True)

# huikang patch 2: manual lm_head LoRA injection
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear
_causal = model
while hasattr(_causal, "model"):
    _causal = _causal.model
_lm = _causal.lm_head
if not isinstance(_lm, LoraLinear):
    model.base_model._create_and_replace(
        LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT),
        "default",
        target=_lm, target_name="lm_head", parent=_causal,
    )
    print("[unsloth-w4] lm_head LoRA: manually injected", flush=True)

# huikang patch 3: fp32 LoRA params
n_fp32 = 0
for name, p in model.named_parameters():
    if ".lora_" in name:
        p.data = p.data.to(torch.float32)
        n_fp32 += 1
print(f"[unsloth-w4] cast {n_fp32} LoRA tensors to fp32", flush=True)
model.print_trainable_parameters()

# ── Data ─────────────────────────────────────────────────────────────
print(f"[unsloth-w4] loading {TRAIN_JSONL}", flush=True)
records = []
with open(TRAIN_JSONL) as f:
    for line in f:
        r = json.loads(line)
        records.append({"prompt": r["prompt_rendered"], "completion": r["completion"]})
print(f"[unsloth-w4] {len(records)} rows", flush=True)
dataset = HFDataset.from_list(records)

# ── SFTConfig (huikang literal) ─────────────────────────────────────
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
os.environ.setdefault("WANDB_RUN_NAME", WANDB_RUN_NAME)
steps_kw = {"max_steps": MAX_STEPS} if MAX_STEPS > 0 else {}
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    **steps_kw,
    per_device_train_batch_size=PER_DEV_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="linear",
    warmup_steps=0,
    max_length=MAX_SEQ_LEN,
    adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
    weight_decay=0.0, max_grad_norm=1e9,
    logging_steps=1,
    save_strategy="steps",
    save_steps=SAVE_EVERY,
    save_total_limit=20,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    remove_unused_columns=False,
    seed=SEED,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    packing=False,
    completion_only_loss=True,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
print(f"[unsloth-w4] starting train: epochs={EPOCHS}, lr={LR}, global_batch=32", flush=True)
t1 = time.time()
result = trainer.train()
wall = (time.time() - t1) / 60.0
final_loss = getattr(result, "metrics", {}).get("train_loss", "?")
print(f"[unsloth-w4] train done in {wall:.1f} min, final_loss={final_loss}", flush=True)

final_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"[unsloth-w4] saved final adapter → {final_dir}", flush=True)
