"""8-GPU DDP Unsloth training entrypoint — huikang 0.85-LB recipe.

Invoked inside a k8s pod via `accelerate launch`. Designed to be driven
entirely by env vars so the same container image services many runs.

Recipe mirrors end-to-end-finetuning-for-lb-0-85 (huikang):
  - LoRA r=32, alpha=32, dropout=0.0, 9 target modules incl. lm_head
  - lr=2e-4 linear, warmup=0, grad_clip=1e9
  - AdamW(β=(0.9, 0.95), ε=1e-8), weight_decay=0.0
  - bf16 base, fp32 LoRA, Mamba fast path forced on
  - completion_only_loss=True via dict rows {prompt, completion}
  - ddp_find_unused_parameters=True (MoE experts + lm_head LoRA)

Env vars:
  MODEL_PATH        (default /fsx/models/Nemotron-3-Nano-30B-A3B-unsloth)
  TRAIN_JSONL       (required — JSONL with `prompt_rendered` + `completion`)
  OUTPUT_DIR        (required — adapter + tokenizer written here)
  MAX_SEQ_LEN       (default 8192)
  PER_DEV_BS        (default 1)
  GRAD_ACCUM        (default 4)      # 8 GPU × 1 × 4 = global 32, matches huikang
  MAX_STEPS         (default 200)    # 0 means use EPOCHS instead
  EPOCHS            (default 1)
  LR                (default 2e-4)
  WANDB_PROJECT     (default nemo-mas-w)
  WANDB_RUN_NAME    (default "unsloth-huikang-${HOSTNAME}")
  SAVE_EVERY_STEPS  (default 50)
"""
from __future__ import annotations
import unsloth  # noqa: F401  (must come before trl/transformers for patches)
from unsloth import FastLanguageModel

import json
import os
import sys
import time

# Inside the pod image Unsloth's ``_unsloth_pre_compute_loss`` fails
# because ``self._old_compute_loss`` was never bound. Provide a minimal
# from-scratch implementation matching what trl 0.24's SFTTrainer.compute_loss
# calls via super(): model forward → take loss from outputs. All the
# Trainer-level plumbing (label smoother, compute_loss_func) doesn't apply
# here because SFTTrainer passes labels in inputs and doesn't set those.
def _install_stock_trainer_methods():
    """Override Unsloth's Trainer patches with stock transformers bodies.

    Must be called AFTER ``FastLanguageModel.from_pretrained`` has run, because
    that call re-applies ``patch_gradient_accumulation_fix`` which overwrites
    Trainer.compute_loss / training_step with wrappers that reference
    ``self._old_*`` attrs that Unsloth never bound on this image's code path.
    Calling the stock bodies directly avoids the whole wrapper chain — their
    numerics (Gemma, grad-accum) don't apply to Nemotron-H SFT.
    """
    from transformers import Trainer as _Tr

    def _min_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            inputs = {**inputs, "num_items_in_batch": num_items_in_batch}
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss = loss * (self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu)
        return (loss, outputs) if return_outputs else loss

    def _min_training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if getattr(self, "use_apex", False):
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        return loss.detach()

    def _min_get_batch_samples(self, epoch_iterator, num_batches, device=None):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        return batch_samples, num_items_in_batch

    _Tr.compute_loss = _min_compute_loss
    _Tr.training_step = _min_training_step
    _Tr.get_batch_samples = _min_get_batch_samples
    _Tr._old_compute_loss = _min_compute_loss
    _Tr._old_training_step = _min_training_step
    # Also unwrap any subclass (SFTTrainer etc.) that inherited from Trainer
    # before we overrode the base — the UnslothSFTTrainer compiled cache
    # inherits via super() so re-binding the base is sufficient.
    print("[unsloth-diag] installed minimal stock Trainer methods (post-FLM)", flush=True)

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from datasets import Dataset as HFDataset
from trl import SFTConfig, SFTTrainer


MODEL_PATH = os.environ.get("MODEL_PATH", "/fsx/models/Nemotron-3-Nano-30B-A3B-unsloth")
TRAIN_JSONL = os.environ["TRAIN_JSONL"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 8192))
PER_DEV_BS = int(os.environ.get("PER_DEV_BS", 1))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 4))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 200))
EPOCHS = int(os.environ.get("EPOCHS", 1))
LR = float(os.environ.get("LR", 2e-4))
SEED = int(os.environ.get("SEED", 42))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY_STEPS", 50))

LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj",
    "in_proj", "out_proj",
    "lm_head",
]


def main() -> None:
    acc = Accelerator()
    rank, world = acc.process_index, acc.num_processes

    def log(msg: str) -> None:
        acc.print(f"[rank {rank}/{world}] {msg}", flush=True)

    set_seed(SEED)
    torch.cuda.set_device(acc.local_process_index)

    # Per-rank Triton/Inductor cache dirs — sharing one dir across 8 ranks
    # races on the same autotune result files.
    _local = acc.local_process_index
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton-unsloth-r{_local}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor-unsloth-r{_local}"

    t0 = time.time()
    log(f"loading base model: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map={"": acc.local_process_index},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"model loaded in {time.time()-t0:.1f}s")

    log("wrapping LoRA (huikang recipe)")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
    )
    FastLanguageModel.for_training(model)

    # Install stock Trainer methods AFTER Unsloth has finished patching
    # transformers.Trainer internals. Otherwise Unsloth overwrites our
    # overrides and the first training step dies on NoneType at _old_*.
    _install_stock_trainer_methods()

    # Mamba fast path (huikang patch #1).
    # On pod images where Triton's SASS compiler produces kernels the
    # cluster driver can't load ("device kernel image is invalid"), set
    # UNSLOTH_DISABLE_MAMBA_KERNELS=1 to fall back to the HF pure-Python
    # SSM loop. ~50-100× slower on Mamba layers but works.
    _nemo = None
    for nm, m in sys.modules.items():
        if "modeling_nemotron_h" in nm and hasattr(m, "is_fast_path_available"):
            _nemo = m
            break
    if _nemo is not None:
        if os.environ.get("UNSLOTH_DISABLE_MAMBA_KERNELS") == "1":
            _nemo.is_fast_path_available = False
            log("Mamba fast path: OFF (env override)")
        else:
            _nemo.is_fast_path_available = True
            log("Mamba fast path: ON")
    else:
        log("WARN: modeling_nemotron_h not patched — will be slow")

    # lm_head LoRA (huikang patch #2).
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
        log("lm_head LoRA manually injected")

    # fp32 LoRA cast (huikang patch #3).
    n_fp32 = 0
    for name, p in model.named_parameters():
        if ".lora_" in name:
            p.data = p.data.to(torch.float32)
            n_fp32 += 1
    log(f"fp32-cast {n_fp32} LoRA tensors")

    # Dataset: JSONL with per-row `prompt_rendered` + `completion`.
    records = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            r = json.loads(line)
            records.append({"prompt": r["prompt_rendered"], "completion": r["completion"]})
    log(f"loaded {len(records)} training rows from {TRAIN_JSONL}")
    dataset = HFDataset.from_list(records)

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
        save_total_limit=10,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=SEED,
        report_to="wandb" if os.environ.get("WANDB_DISABLED", "").lower() != "true" else "none",
        packing=False,
        ddp_find_unused_parameters=True,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    effective_batch = PER_DEV_BS * GRAD_ACCUM * world
    log(
        f"starting train: effective_batch={effective_batch}, "
        f"max_steps={MAX_STEPS}, epochs={EPOCHS}, lr={LR}"
    )
    train_start = time.time()
    result = trainer.train()
    log(
        f"train done in {(time.time()-train_start)/60.0:.1f} min — "
        f"final_loss={getattr(result,'metrics',{}).get('train_loss','?')}"
    )

    # Rank-0 only: final adapter save.
    if acc.is_main_process:
        final_dir = os.path.join(OUTPUT_DIR, "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        log(f"saved final adapter → {final_dir}")
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
