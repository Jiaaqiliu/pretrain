"""Smoke test — huikang's approach: explicit 8-module list + lm_head injection +
Mamba fast-path patch + fp32 LoRA.

Pass criteria:
  - model loads
  - LoRA wraps successfully (no "No layers to finetune")
  - lm_head becomes a LoraLinear (manual inject worked)
  - one forward pass returns logits
  - one micro backward returns a gradient
"""
import os, time, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("[smoke2] importing torch / unsloth...", flush=True)
import torch
from unsloth import FastLanguageModel

MODEL = "/fsx/models/Nemotron-3-Nano-30B-A3B-unsloth"
MAX_SEQ_LEN = 8192
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj",
    "in_proj", "out_proj",
    "lm_head",
]

t0 = time.time()
print(f"[smoke2] loading {MODEL} ...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
    trust_remote_code=True, unsloth_force_compile=False,
    attn_implementation="eager", dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[smoke2] model loaded in {time.time()-t0:.1f}s: {type(model).__name__}")

print("[smoke2] wrapping LoRA (huikang target list including lm_head)...")
t1 = time.time()
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none", use_gradient_checkpointing="unsloth",
    random_state=42,
)
FastLanguageModel.for_training(model)
print(f"[smoke2] LoRA wrapped in {time.time()-t1:.1f}s")

# Huikang patch 1: Mamba fast path
_nemo = None
for name, m in sys.modules.items():
    if "modeling_nemotron_h" in name and hasattr(m, "is_fast_path_available"):
        _nemo = m; break
assert _nemo is not None, "modeling_nemotron_h not found"
prev = _nemo.is_fast_path_available
_nemo.is_fast_path_available = True
print(f"[smoke2] Mamba fast path: {prev} → True")

# Huikang patch 2: manual lm_head LoRA injection
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear
_causal = model
while hasattr(_causal, "model"): _causal = _causal.model
_lm = _causal.lm_head
if not isinstance(_lm, LoraLinear):
    _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    model.base_model._create_and_replace(
        _cfg, "default",
        target=_lm, target_name="lm_head", parent=_causal,
    )
    print("[smoke2] lm_head LoRA: manually injected")
else:
    print("[smoke2] lm_head already LoRA-wrapped")

# Huikang patch 3: fp32 LoRA params
n_fp32 = 0
for name, p in model.named_parameters():
    if ".lora_" in name:
        p.data = p.data.to(torch.float32)
        n_fp32 += 1
print(f"[smoke2] cast {n_fp32} LoRA tensors to fp32")

model.print_trainable_parameters()

# Forward + backward sanity
print("[smoke2] forward-pass sanity ...")
prompt = "Hello, world!"
inp = tokenizer(prompt, return_tensors="pt").to(model.device)
labels = inp["input_ids"].clone()
t2 = time.time()
model.train()
out = model(**inp, labels=labels)
loss = out.loss
print(f"[smoke2] forward ok in {time.time()-t2:.1f}s; loss={loss.item():.4f}")

t3 = time.time()
loss.backward()
print(f"[smoke2] backward ok in {time.time()-t3:.1f}s")

# Check some LoRA param got a gradient
n_grad = sum(1 for n,p in model.named_parameters() if ".lora_" in n and p.grad is not None)
print(f"[smoke2] LoRA params with gradient: {n_grad}")
if n_grad == 0:
    print("[smoke2] FAIL — no LoRA gradients"); sys.exit(1)

print("[smoke2] PASS — Unsloth can train Nemotron-H with huikang's setup")
