"""Smoke-test: does Unsloth actually load Nemotron-3-Nano-30B-A3B?

If yes, advance to Stage 2 (multi-GPU) and Stage 3 (k8s).
If no (e.g. "architecture not supported"), we know to stop.
"""
import os, time, sys

# Pin to single GPU 1 (avoid the stray process on GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("[smoke] importing torch / unsloth...", flush=True)
import torch
from unsloth import FastLanguageModel

MODEL = "/fsx/models/Nemotron-3-Nano-30B-A3B-unsloth"
t0 = time.time()
print(f"[smoke] loading {MODEL} with Unsloth FastLanguageModel...", flush=True)
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=8192,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s")
    print(f"[smoke] model class: {type(model).__name__}")
    print(f"[smoke] tokenizer class: {type(tokenizer).__name__}")
except Exception as e:
    print(f"[smoke] FAILED to load model: {type(e).__name__}: {e}")
    sys.exit(1)

print("[smoke] wrapping LoRA (W4 literal: r=32, α=32, target=all-linear)...")
t1 = time.time()
try:
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"[smoke] LoRA wrapped in {time.time()-t1:.1f}s")
    model.print_trainable_parameters()
except Exception as e:
    print(f"[smoke] FAILED to wrap LoRA: {type(e).__name__}: {e}")
    sys.exit(2)

# Mini forward pass to confirm the lora-wrapped model runs
print("[smoke] forward-pass sanity (1 tiny prompt)...")
prompt = "Hello, world!"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
t2 = time.time()
with torch.no_grad():
    out = model(**inputs)
print(f"[smoke] forward ok in {time.time()-t2:.1f}s; output logits shape {out.logits.shape}")
print("[smoke] PASS — Unsloth can load + LoRA + forward on Nemotron")
