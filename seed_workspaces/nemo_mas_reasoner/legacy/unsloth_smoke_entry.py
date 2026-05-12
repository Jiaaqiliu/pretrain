"""Single-GPU Unsloth training smoke — 10 opt steps, tiny corpus.

Used to validate whether 8 concurrent instances share a single node
cleanly. Each pod reads LR from env (for a mini sweep), writes outputs
to its own OUTPUT_DIR, loads no shared cache.

Env:
    LR              — learning rate (e.g. 1e-4, 2e-4, 5e-4)
    RANK_IDX        — 0..7, for stamping output dir
    OUTPUT_DIR      — where to save the tiny adapter + loss log
    MODEL_PATH      — model dir
    TRAIN_JSONL     — jsonl with prompt_rendered + completion
    N_ROWS          — how many rows to tokenize (default 200)
    NUM_STEPS       — opt steps (default 10)
"""
from __future__ import annotations
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import gc, json, math, os, sys, time
import torch
from cut_cross_entropy import linear_cross_entropy
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear

MODEL_PATH  = os.environ.get("MODEL_PATH", "/fsx/models/Nemotron-3-Nano-30B-A3B-unsloth")
TRAIN_JSONL = os.environ.get("TRAIN_JSONL",
    "/fsx/zzsamshi/a-evolve/runs/nemo-mas-teams-v2/cycles/0002/.fork_target/nodes/workspace/workspace/data/final/train.jsonl")
OUTPUT_DIR  = os.environ["OUTPUT_DIR"]
RANK_IDX    = int(os.environ.get("RANK_IDX", 0))
LR          = float(os.environ.get("LR", 2e-4))
N_ROWS      = int(os.environ.get("N_ROWS", 200))
NUM_STEPS   = int(os.environ.get("NUM_STEPS", 10))
SEED        = int(os.environ.get("SEED", 42 + RANK_IDX))

MAX_SEQ_LEN      = 8192
LORA_RANK        = 32
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.0
BATCH_SIZE       = 8   # smaller for smoke
MICRO_BATCH_SIZE = 2
TARGET_MODULES   = [
    "q_proj","k_proj","v_proj","o_proj",
    "up_proj","down_proj","in_proj","out_proj","lm_head",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, "smoke.log")
def log(msg):
    line = f"[rank {RANK_IDX} lr={LR:.1e}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

log(f"start: GPU={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
log(f"torch={torch.__version__} cuda={torch.version.cuda}")

import causal_conv1d, mamba_ssm
log(f"mamba_ssm={mamba_ssm.__version__} causal_conv1d={causal_conv1d.__version__}")

# Sanity: load tiny mamba kernel
from causal_conv1d import causal_conv1d_fn
_x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
_w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
causal_conv1d_fn(_x, _w, None, activation="silu")
log("mamba kernel smoke ok")

gc.collect(); torch.cuda.empty_cache()

t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False, load_in_8bit=False,
    full_finetuning=False, trust_remote_code=True,
    unsloth_force_compile=True,
    attn_implementation="eager", dtype=torch.bfloat16,
)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
log(f"model loaded in {time.time()-t0:.1f}s")

model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK, target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
FastLanguageModel.for_training(model)

# Mamba fast path
nemotron_mod = None
for _name, _m in sys.modules.items():
    if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
        nemotron_mod = _m; break
assert nemotron_mod is not None
nemotron_mod.is_fast_path_available = True
log("mamba fast path ON")

# lm_head LoRA
_c = model
while hasattr(_c, "model"): _c = _c.model
if not isinstance(_c.lm_head, LoraLinear):
    model.base_model._create_and_replace(
        LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT),
        "default", target=_c.lm_head, target_name="lm_head", parent=_c,
    )
    log("lm_head LoRA injected")

for name, p in model.named_parameters():
    if ".lora_" in name: p.data = p.data.to(torch.float32)
log("LoRA fp32 cast done")

# CCE patched forward
_base = model
while hasattr(_base, "model"): _base = _base.model
def _patched(input_ids=None, attention_mask=None, labels=None, **kwargs):
    o = _base.backbone(input_ids=input_ids, attention_mask=attention_mask,
        **{k:v for k,v in kwargs.items() if k in ("position_ids","past_key_values","use_cache")})
    h = o[0]; lm = _base.lm_head
    W = lm.base_layer.weight + lm.scaling["default"] * lm.lora_B["default"].weight @ lm.lora_A["default"].weight
    ce = linear_cross_entropy(h, W, labels, reduction="none") if labels is not None else None
    model._cached_per_token_ce = ce
    return ce.mean() if ce is not None else None
_base.forward = _patched

# MoE tying
moe_tied_params = []
for name, p in model.named_parameters():
    if not p.requires_grad or ".experts." not in name or ".lora_" not in name: continue
    is_w1 = any(s in name for s in ("gate_up_proj","up_proj","gate_proj",".w1."))
    is_w2 = any(s in name for s in ("down_proj",".w2."))
    is_A = ".lora_A." in name; is_B = ".lora_B." in name
    if ((is_w1 and is_A) or (is_w2 and is_B)) and p.dim() >= 2 and p.shape[0] > 1:
        moe_tied_params.append(p)
def tie_init():
    with torch.no_grad():
        for p in moe_tied_params:
            m = p.data.mean(dim=0, keepdim=True); p.data.copy_(m.expand_as(p.data))
def tie_grads():
    with torch.no_grad():
        for p in moe_tied_params:
            if p.grad is None: continue
            g = p.grad.sum(dim=0, keepdim=True); p.grad.copy_(g.expand_as(p.grad))
tie_init()
log(f"MoE tied params: {len(moe_tied_params)}")

# Load tiny corpus
examples = []
with open(TRAIN_JSONL) as f:
    for i, line in enumerate(f):
        if i >= N_ROWS: break
        r = json.loads(line)
        prompt_ids = tokenizer(r["prompt_rendered"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(r["completion"] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        toks = prompt_ids + completion_ids
        mask = [0.0]*len(prompt_ids) + [1.0]*len(completion_ids)
        if len(toks) > MAX_SEQ_LEN: toks=toks[:MAX_SEQ_LEN]; mask=mask[:MAX_SEQ_LEN]
        if not any(mask): continue
        examples.append({"tokens":toks[:-1],"targets":toks[1:],"weights":mask[1:]})
log(f"{len(examples)} examples")

device = next(model.parameters()).device
optimizer = None
step = 0
t_train = time.time()
for bstart in range(0, len(examples), BATCH_SIZE):
    if step >= NUM_STEPS: break
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
            lr=LR, betas=(0.9,0.95), eps=1e-8, weight_decay=0.0)
    tie_grads()
    gn = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1e9)
    optimizer.step(); optimizer.zero_grad()
    lm = total_l/total_w if total_w > 0 else 0
    step += 1
    log(f"step {step}/{NUM_STEPS} loss={lm:.4f} grad={gn:.3f} elapsed={(time.time()-t_train)/60:.2f}min")

log(f"DONE — peak_VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
