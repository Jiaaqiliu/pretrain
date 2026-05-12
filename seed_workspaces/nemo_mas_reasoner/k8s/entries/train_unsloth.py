"""Unsloth SFT on Nemotron-3-Nano-30B-A3B, recipe-driven, single-GPU.

Reads recipe from YAML (default: train/recipes/huikang.yaml). Env vars
override specific knobs (LR, NUM_STEPS, SAVE_EVERY, SEED, RUN_NAME).
Launched inside a k8s pod by jobs/train_1gpu.yaml.

Env:
    RECIPE_PATH     — recipe YAML (default: workspace train/recipes/huikang.yaml)
    OUTPUT_DIR      — where to save step_{N}/ + final/ (required)
    RUN_NAME        — for wandb + log prefix (default 'unsloth-train')
    LR              — override recipe.optimizer.lr
    NUM_STEPS       — override recipe.batching.num_steps
    SAVE_EVERY      — override recipe.batching.save_every
    SEED            — override recipe.batching.seed
"""
from __future__ import annotations
import unsloth  # noqa: F401 — must come first
from unsloth import FastLanguageModel

import gc, json, math, os, sys, time
import torch, yaml
from cut_cross_entropy import linear_cross_entropy
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear

# ── Locate + load recipe ─────────────────────────────────────────────
DEFAULT_RECIPE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "train", "recipes", "huikang.yaml",
)
RECIPE_PATH = os.environ.get("RECIPE_PATH", DEFAULT_RECIPE)
with open(RECIPE_PATH) as _f: R = yaml.safe_load(_f)

# ── Env overrides (knobs the CLI sweeps) ─────────────────────────────
def _env_or(key: str, default, cast):
    v = os.environ.get(key, "")
    return cast(v) if v else cast(default)

OUTPUT_DIR = os.environ["OUTPUT_DIR"]
RUN_NAME   = os.environ.get("RUN_NAME") or f"unsloth-{R['name']}"
LR         = _env_or("LR",         R["optimizer"]["lr"],           float)
NUM_STEPS  = _env_or("NUM_STEPS",  R["batching"]["num_steps"],      int)
SAVE_EVERY = _env_or("SAVE_EVERY", R["batching"]["save_every"],     int)
SEED       = _env_or("SEED",       R["batching"]["seed"],           int)

# ── Fixed recipe values ─────────────────────────────────────────────
MODEL_PATH     = R["model"]["path"]
TRAIN_JSONL    = R["data"]["train_jsonl"]
MAX_SEQ_LEN    = int(R["model"]["max_seq_len"])
LORA_RANK      = int(R["adapter"]["rank"])
LORA_ALPHA     = int(R["adapter"]["alpha"])
LORA_DROPOUT   = float(R["adapter"]["dropout"])
TARGET_MODULES = list(R["adapter"]["target_modules"])
USE_GC         = R["adapter"].get("use_gradient_checkpointing", "unsloth")
BATCH_SIZE       = int(R["batching"]["batch_size"])
MICRO_BATCH_SIZE = int(R["batching"]["micro_batch_size"])
OPT_BETAS      = tuple(R["optimizer"]["betas"])
OPT_EPS        = float(R["optimizer"]["eps"])
OPT_WD         = float(R["optimizer"]["weight_decay"])
GRAD_CLIP      = float(R["optimizer"]["max_grad_norm"])
SCHED_TYPE     = R["scheduler"]["type"]
WARMUP_STEPS   = int(R["scheduler"]["warmup_steps"])
MOE_TIE        = bool(R["tricks"]["moe_tie_weights"])
USE_CCE        = bool(R["tricks"]["cce_patched_forward"])
MAMBA_FAST     = bool(R["tricks"]["mamba_fast_path"])
FORCE_COMPILE  = bool(R["tricks"]["unsloth_force_compile"])
ATTN_IMPL      = R["tricks"].get("attn_implementation", "eager")

os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, "train.log")

def log(msg: str) -> None:
    line = f"[{RUN_NAME}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

log(f"recipe={RECIPE_PATH} ({R['name']})")
log(f"overrides: lr={LR} steps={NUM_STEPS} save_every={SAVE_EVERY} seed={SEED}")
log(f"output={OUTPUT_DIR}")

# ── Sanity: GPU + kernel smoke ───────────────────────────────────────
import causal_conv1d, mamba_ssm
cc = torch.cuda.get_device_capability(0)
log(f"GPU={torch.cuda.get_device_name(0)} sm_{cc[0]*10+cc[1]} "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"mamba_ssm={mamba_ssm.__version__} causal_conv1d={causal_conv1d.__version__}")
from causal_conv1d import causal_conv1d_fn
_x = torch.randn(1,256,32,device="cuda",dtype=torch.bfloat16)
_w = torch.randn(256,4,device="cuda",dtype=torch.bfloat16)
causal_conv1d_fn(_x, _w, None, activation="silu")
log("causal_conv1d kernel OK")

# ── Load base model ──────────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False, load_in_8bit=False,
    full_finetuning=False, trust_remote_code=True,
    unsloth_force_compile=FORCE_COMPILE,
    attn_implementation=ATTN_IMPL, dtype=torch.bfloat16,
)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
log(f"model loaded in {time.time()-t0:.1f}s")

# ── LoRA wrap ────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK, target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", use_gradient_checkpointing=USE_GC,
    random_state=SEED,
)
FastLanguageModel.for_training(model)

# ── Patch 1: Mamba fast path ─────────────────────────────────────────
if MAMBA_FAST:
    nemotron_mod = None
    for _n, _m in sys.modules.items():
        if "modeling_nemotron_h" in _n and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m; break
    assert nemotron_mod is not None
    nemotron_mod.is_fast_path_available = True
    log("Mamba fast path = True")

# ── Patch 2: lm_head LoRA (Unsloth drops it for MoE) ─────────────────
if "lm_head" in TARGET_MODULES:
    _c = model
    while hasattr(_c, "model"): _c = _c.model
    if not isinstance(_c.lm_head, LoraLinear):
        model.base_model._create_and_replace(
            LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT),
            "default", target=_c.lm_head, target_name="lm_head", parent=_c,
        )
        log("lm_head LoRA injected")

# ── Patch 3: fp32 LoRA, assertions for dtype discipline ──────────────
for name, p in model.named_parameters():
    if ".lora_" in name: p.data = p.data.to(torch.float32)
for name, p in model.named_parameters():
    if ".lora_" in name:
        assert p.dtype == torch.float32, f"LoRA {name} wanted fp32 got {p.dtype}"; continue
    if ".mixer.gate." in name:
        assert p.dtype == torch.float32, f"router {name} wanted fp32 got {p.dtype}"; continue
    assert p.dtype == torch.bfloat16, f"{name} wanted bf16 got {p.dtype}"
log("dtypes verified: LoRA fp32, base bf16, MoE router fp32")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"trainable={trainable:,}")

# ── Patch 4: CCE forward ─────────────────────────────────────────────
if USE_CCE:
    _base = model
    while hasattr(_base, "model"): _base = _base.model
    def _patched_forward(input_ids=None, attention_mask=None, labels=None, **kw):
        o = _base.backbone(input_ids=input_ids, attention_mask=attention_mask,
            **{k:v for k,v in kw.items() if k in ("position_ids","past_key_values","use_cache")})
        h = o[0]; lm = _base.lm_head
        W = lm.base_layer.weight + lm.scaling["default"] * lm.lora_B["default"].weight @ lm.lora_A["default"].weight
        ce = linear_cross_entropy(h, W, labels, reduction="none") if labels is not None else None
        model._cached_per_token_ce = ce
        return ce.mean() if ce is not None else None
    _base.forward = _patched_forward
    log("CCE patched forward installed")

# ── MoE weight tying (Tinker convention) ────────────────────────────
moe_tied_params = []
if MOE_TIE:
    w1_names = ("gate_up_proj","up_proj","gate_proj",".w1.")
    w2_names = ("down_proj",".w2.")
    for name, p in model.named_parameters():
        if not p.requires_grad or ".experts." not in name or ".lora_" not in name: continue
        is_w1 = any(s in name for s in w1_names); is_w2 = any(s in name for s in w2_names)
        is_A  = ".lora_A." in name; is_B = ".lora_B." in name
        if ((is_w1 and is_A) or (is_w2 and is_B)) and p.dim() >= 2 and p.shape[0] > 1:
            moe_tied_params.append(p)
    with torch.no_grad():
        for p in moe_tied_params:
            m = p.data.mean(dim=0, keepdim=True); p.data.copy_(m.expand_as(p.data))
    log(f"MoE tying: {len(moe_tied_params)} tensors tied")

def tie_grads():
    with torch.no_grad():
        for p in moe_tied_params:
            if p.grad is None: continue
            g = p.grad.sum(dim=0, keepdim=True); p.grad.copy_(g.expand_as(p.grad))

# ── Tokenize corpus ──────────────────────────────────────────────────
log(f"tokenizing {TRAIN_JSONL}")
examples = []
t1 = time.time()
with open(TRAIN_JSONL) as f:
    for line in f:
        r = json.loads(line)
        prompt_ids     = tokenizer(r["prompt_rendered"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(r["completion"] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        toks = prompt_ids + completion_ids
        mask = [0.0]*len(prompt_ids) + [1.0]*len(completion_ids)
        if len(toks) > MAX_SEQ_LEN: toks = toks[:MAX_SEQ_LEN]; mask = mask[:MAX_SEQ_LEN]
        if not any(mask): continue
        examples.append({"tokens": toks[:-1], "targets": toks[1:], "weights": mask[1:]})
log(f"tokenized {len(examples)} examples in {time.time()-t1:.1f}s")

max_steps = len(examples) // BATCH_SIZE
num_steps = min(NUM_STEPS, max_steps)
log(f"training: {num_steps} steps (clamped from NUM_STEPS={NUM_STEPS}, max={max_steps})")

# ── Training loop ────────────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
device = next(model.parameters()).device
optimizer = None
step = 0
t_train = time.time()
for bstart in range(0, len(examples), BATCH_SIZE):
    if step >= num_steps: break
    batch = examples[bstart:bstart+BATCH_SIZE]
    n = len(batch); n_accum = math.ceil(n/MICRO_BATCH_SIZE)
    total_l = 0.0; total_w = 0.0
    for mbs in range(0, n, MICRO_BATCH_SIZE):
        mb = batch[mbs:mbs+MICRO_BATCH_SIZE]
        nm = len(mb); ml = max(len(e["tokens"]) for e in mb)
        pi = torch.zeros(nm, ml, dtype=torch.long, device=device)
        pt = torch.zeros(nm, ml, dtype=torch.long, device=device)
        pw = torch.zeros(nm, ml, dtype=torch.float32, device=device)
        am = torch.zeros(nm, ml, dtype=torch.long, device=device)
        for i, e in enumerate(mb):
            sl = len(e["tokens"])
            pi[i,:sl] = torch.tensor(e["tokens"])
            pt[i,:sl] = torch.tensor(e["targets"])
            pw[i,:sl] = torch.tensor(e["weights"], dtype=torch.float32)
            am[i,:sl] = 1
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=pi, attention_mask=am, labels=pt, use_cache=False)
            ce = model._cached_per_token_ce
            ws = pw.sum(); ls = (ce*pw).sum()
            loss = ls/ws if ws > 0 else ls*0.0
        (loss/n_accum).backward()
        total_l += ls.item(); total_w += ws.item()
        del loss, ce
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR, betas=OPT_BETAS, eps=OPT_EPS, weight_decay=OPT_WD)
    # scheduler
    if SCHED_TYPE == "linear":
        if step < WARMUP_STEPS:
            lr = LR * (step + 1) / max(1, WARMUP_STEPS)
        else:
            denom = max(1, num_steps - WARMUP_STEPS)
            lr = LR * max(0.0, (num_steps - step) / denom)
    else:
        lr = LR
    for pg in optimizer.param_groups: pg["lr"] = lr
    tie_grads()
    gn = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=GRAD_CLIP)
    optimizer.step(); optimizer.zero_grad()
    lm = total_l/total_w if total_w > 0 else 0
    step += 1
    log(f"step {step}/{num_steps} loss={lm:.6f} grad={gn:.4f} lr={lr:.2e} "
        f"elapsed={(time.time()-t_train)/60:.1f}min")
    if step % SAVE_EVERY == 0 or step == num_steps:
        sd = os.path.join(OUTPUT_DIR, f"step_{step}")
        os.makedirs(sd, exist_ok=True)
        model.save_pretrained(sd); tokenizer.save_pretrained(sd)
        log(f"saved {sd}")

# ── Final adapter + Kaggle-compat lm_head key rename ─────────────────
from safetensors.torch import load_file, save_file
final_dir = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir); tokenizer.save_pretrained(final_dir)
st_path = os.path.join(final_dir, "adapter_model.safetensors")
tensors = load_file(st_path)
renamed = {k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
           for k, v in tensors.items()}
save_file(renamed, st_path)
log(f"DONE — final={final_dir} peak_VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
