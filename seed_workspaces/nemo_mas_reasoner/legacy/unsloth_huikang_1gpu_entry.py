"""Huikang's 0.85-LB recipe — faithful 1-GPU reproduction.

Ports everything from huikang's Kaggle notebook
(end-to-end-finetuning-for-lb-0-85.ipynb) EXCEPT the corpus loader:
huikang uses his private pre-tokenized `tokens`+`mask` dataset; we
derive the same (tokens, weights-mask) from our `prompt_rendered` +
`completion` JSONL by re-tokenizing. Everything downstream — custom
training loop, Cut Cross-Entropy forward, MoE weight tying, tied-grad
sum, linear LR decay, save-rename — is identical.

Launch:
    CUDA_VISIBLE_DEVICES=7 python /tmp/unsloth_huikang_1gpu.py
"""
from __future__ import annotations
import unsloth  # noqa: F401 — must come first
from unsloth import FastLanguageModel

import gc, json, math, os, random, sys, time
import torch
from cut_cross_entropy import linear_cross_entropy
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear

# ── Config (huikang literal) ─────────────────────────────────────────
LORA_RANK         = 32
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.0
MAX_SEQ_LEN       = 8192
NUM_STEPS         = 1000            # will clamp to max_steps = examples // BATCH_SIZE
BATCH_SIZE        = 32
MICRO_BATCH_SIZE  = 4
LEARNING_RATE     = 2e-4
RESET_WEIGHTS     = True
IN_PROJ_ONLY      = False
MOE_TIE_WEIGHTS   = True
SHUFFLE_DATASET   = False
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj",
    "in_proj", "out_proj",
    "lm_head",
]

MODEL_PATH  = "/fsx/models/Nemotron-3-Nano-30B-A3B-unsloth"
TRAIN_JSONL = "/fsx/zzsamshi/a-evolve/runs/nemo-mas-teams-v2/cycles/0002/.fork_target/nodes/workspace/workspace/data/final/train.jsonl"
OUTPUT_DIR  = "/fsx/zzsamshi/a-evolve/runs/nemo-mas-teams-v2/cycles/0002/.fork_target/nodes/workspace/workspace/checkpoints/adapters/sft_w4_unsloth_huikang_k8s"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sanity print ─────────────────────────────────────────────────────
import causal_conv1d, mamba_ssm
cc = torch.cuda.get_device_capability(0)
print(f"[huikang] GPU: {torch.cuda.get_device_name(0)} sm_{cc[0]*10+cc[1]}", flush=True)
print(f"[huikang] torch={torch.__version__}, cuda={torch.version.cuda}", flush=True)
print(f"[huikang] mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}", flush=True)
from causal_conv1d import causal_conv1d_fn
_x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
_w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
causal_conv1d_fn(_x, _w, None, activation="silu")
print("[huikang] causal_conv1d CUDA kernel: OK", flush=True)

# ── Load base model ──────────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False, load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=True,
    attn_implementation="eager",
    dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[huikang] model loaded in {time.time()-t0:.1f}s", flush=True)

# ── LoRA wrap ────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK, target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
FastLanguageModel.for_training(model)

# ── Mamba fast path ──────────────────────────────────────────────────
nemotron_mod = None
for _name, _m in sys.modules.items():
    if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
        nemotron_mod = _m; break
assert nemotron_mod is not None
nemotron_mod.is_fast_path_available = True
print("[huikang] patched is_fast_path_available = True", flush=True)

# ── Manual lm_head LoRA ──────────────────────────────────────────────
_causal_lm = model
while hasattr(_causal_lm, "model"):
    _causal_lm = _causal_lm.model
_lm_head = _causal_lm.lm_head
if not isinstance(_lm_head, LoraLinear):
    _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    model.base_model._create_and_replace(
        _cfg, "default", target=_lm_head, target_name="lm_head", parent=_causal_lm,
    )
    print("[huikang] lm_head LoRA injected", flush=True)

# ── Dtype cast + strict assertions ───────────────────────────────────
for name, param in model.named_parameters():
    if ".lora_" in name:
        param.data = param.data.to(torch.float32)
for name, param in model.named_parameters():
    if ".lora_" in name:
        assert param.dtype == torch.float32, f"LoRA {name} expected fp32, got {param.dtype}"
        continue
    is_router = ".mixer.gate." in name
    if is_router:
        assert param.dtype == torch.float32, f"router {name} expected fp32, got {param.dtype}"
        continue
    assert param.dtype == torch.bfloat16, f"{name} expected bf16, got {param.dtype}"
print("[huikang] verified LoRA fp32, base bf16, MoE router fp32", flush=True)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"[huikang] {trainable:,} trainable / {total:,} total", flush=True)

# ── Patch CausalLM.forward with Cut Cross-Entropy ───────────────────
_base = model
while hasattr(_base, "model"):
    _base = _base.model

def _patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
    backbone_out = _base.backbone(
        input_ids=input_ids, attention_mask=attention_mask,
        **{k: v for k, v in kwargs.items() if k in ("position_ids", "past_key_values", "use_cache")},
    )
    hidden_states = backbone_out[0]
    lm_head = _base.lm_head
    base_w = lm_head.base_layer.weight
    lora_A = lm_head.lora_A["default"].weight
    lora_B = lm_head.lora_B["default"].weight
    scaling = lm_head.scaling["default"]
    lm_weight = base_w + scaling * lora_B @ lora_A
    if labels is not None:
        per_token_ce = linear_cross_entropy(hidden_states, lm_weight, labels, reduction="none")
        loss = per_token_ce.mean()
    else:
        per_token_ce = None; loss = None
    model._cached_per_token_ce = per_token_ce
    return loss

_base.forward = _patched_causal_forward
print("[huikang] patched CausalLM.forward with CCE (no logits materialization)", flush=True)

# ── MoE weight tying (Tinker convention) ────────────────────────────
moe_tied_params: list[torch.Tensor] = []
if MOE_TIE_WEIGHTS:
    w1_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
    w2_names = ("down_proj", ".w2.")
    for name, param in model.named_parameters():
        if not param.requires_grad or ".experts." not in name or ".lora_" not in name:
            continue
        is_w1 = any(p in name for p in w1_names)
        is_w2 = any(p in name for p in w2_names)
        is_A = ".lora_A." in name
        is_B = ".lora_B." in name
        should_tie = (is_w1 and is_A) or (is_w2 and is_B)
        if not should_tie:
            continue
        if param.dim() < 2 or param.shape[0] <= 1:
            continue
        moe_tied_params.append(param)

    def _tie_init():
        with torch.no_grad():
            for p in moe_tied_params:
                mean = p.data.mean(dim=0, keepdim=True)
                p.data.copy_(mean.expand_as(p.data))

    def _tie_grads():
        with torch.no_grad():
            for p in moe_tied_params:
                if p.grad is None: continue
                grad_sum = p.grad.sum(dim=0, keepdim=True)
                p.grad.copy_(grad_sum.expand_as(p.grad))
    print(f"[huikang] MoE tying: {len(moe_tied_params)} tensors tied", flush=True)
    _tie_init()
else:
    def _tie_grads(): pass

# ── Corpus: tokenize (prompt, completion) → tokens + per-token weights ──
# Huikang's `tokens`/`mask` format: tokens[1:] are targets, mask[1:] is
# 1.0 on completion tokens and 0.0 on prompt tokens.
print(f"[huikang] tokenizing {TRAIN_JSONL}", flush=True)
examples: list[dict] = []
t1 = time.time()
with open(TRAIN_JSONL) as f:
    for line in f:
        r = json.loads(line)
        prompt = r["prompt_rendered"]
        completion = r["completion"] + (tokenizer.eos_token or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        tokens = prompt_ids + completion_ids
        mask = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]; mask = mask[:MAX_SEQ_LEN]
        if not any(mask): continue
        examples.append({
            "problem_id": r.get("id", ""),
            "tokens":  tokens[:-1],
            "targets": tokens[1:],
            "weights": mask[1:],
        })
print(f"[huikang] {len(examples)} examples in {time.time()-t1:.1f}s", flush=True)
total_unmasked = sum(sum(e["weights"]) for e in examples)
total_tokens = sum(len(e["tokens"]) for e in examples)
print(f"[huikang] {total_tokens:,} tokens total, {total_unmasked:,.0f} unmasked", flush=True)

# ── Training loop (huikang literal) ──────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
device = next(model.parameters()).device

indices = list(range(len(examples)))
if SHUFFLE_DATASET:
    random.Random(0).shuffle(indices)

max_steps = len(examples) // BATCH_SIZE
num_steps = min(NUM_STEPS, max_steps)
print(f"[huikang] training: {num_steps} steps (clamped from {NUM_STEPS} "
      f"vs max {max_steps}), batch={BATCH_SIZE}, micro={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}", flush=True)

optimizer: torch.optim.AdamW | None = None
training_log: list[str] = []
step = 0
t_train = time.time()
for batch_start in range(0, len(indices), BATCH_SIZE):
    if step >= num_steps: break
    batch_idx = indices[batch_start: batch_start + BATCH_SIZE]
    batch = [examples[i] for i in batch_idx]
    batch_tokens  = [e["tokens"]  for e in batch]
    batch_targets = [e["targets"] for e in batch]
    batch_weights = [e["weights"] for e in batch]

    n = len(batch)
    n_accum = math.ceil(n / MICRO_BATCH_SIZE)
    total_loss_sum = 0.0
    total_weight_sum = 0.0

    for mb_start in range(0, n, MICRO_BATCH_SIZE):
        mb_end = min(mb_start + MICRO_BATCH_SIZE, n)
        mb_toks = batch_tokens[mb_start:mb_end]
        mb_tgts = batch_targets[mb_start:mb_end]
        mb_wts  = batch_weights[mb_start:mb_end]
        n_micro = len(mb_toks)
        max_len = max(len(t) for t in mb_toks)
        padded_input   = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
        padded_targets = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
        padded_weights = torch.zeros(n_micro, max_len, dtype=torch.float32, device=device)
        attention_mask = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
        for i in range(n_micro):
            seq_len = len(mb_toks[i])
            padded_input  [i, :seq_len] = torch.tensor(mb_toks[i], dtype=torch.long)
            padded_targets[i, :seq_len] = torch.tensor(mb_tgts[i], dtype=torch.long)
            padded_weights[i, :seq_len] = torch.tensor(mb_wts[i],  dtype=torch.float32)
            attention_mask[i, :seq_len] = 1
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=padded_input, attention_mask=attention_mask,
                  labels=padded_targets, use_cache=False)
            per_token_ce = model._cached_per_token_ce
            weighted_loss = per_token_ce * padded_weights
            w_sum = padded_weights.sum()
            l_sum = weighted_loss.sum()
            loss = l_sum / w_sum if w_sum > 0 else l_sum * 0.0
        (loss / n_accum).backward()
        total_loss_sum   += l_sum.item()
        total_weight_sum += w_sum.item()
        del loss, per_token_ce, weighted_loss

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        )
    lr = LEARNING_RATE * (1 - step / num_steps)
    for pg in optimizer.param_groups: pg["lr"] = lr
    _tie_grads()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1e9)
    optimizer.step(); optimizer.zero_grad()
    loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0
    step += 1
    elapsed = (time.time() - t_train) / 60.0
    msg = (f"[huikang] step {step}/{num_steps}: loss={loss_mean:.6f} "
           f"grad={grad_norm:.4f} lr={lr:.2e} elapsed={elapsed:.1f}min")
    print(msg, flush=True); training_log.append(msg)

    # Save every 50 steps
    if step % 50 == 0 or step == num_steps:
        save_dir = os.path.join(OUTPUT_DIR, f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[huikang] saved → {save_dir}", flush=True)

# ── Save final + rename lm_head keys for Kaggle submission ──────────
from safetensors.torch import load_file, save_file
final_dir = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
st_path = os.path.join(final_dir, "adapter_model.safetensors")
tensors = load_file(st_path)
renamed = {
    k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
    for k, v in tensors.items()
}
save_file(renamed, st_path)
with open(os.path.join(final_dir, "training_log.txt"), "w") as f:
    f.write("\n".join(training_log) + "\n")
print(f"[huikang] DONE — final adapter at {final_dir}, "
      f"peak VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
