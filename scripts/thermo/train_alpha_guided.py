"""Train with α-guided adaptive LR schedule.

Compares: Cosine (baseline) vs α-Guided (constant LR → decay at α reversal).

The α-guided schedule:
1. Warmup (2% of steps)
2. Constant peak LR (until α reversal detected)
3. Linear decay to min_lr (from reversal to end)

Fallback: if no reversal by 80% of training, force start decay.

Usage:
    torchrun --nproc_per_node=8 scripts/thermo/train_alpha_guided.py \
        --schedule cosine --seed 42 \
        --output /fsx/dev/jiaqi/thermo_results/alpha_guided/cosine_s42.jsonl

    torchrun --nproc_per_node=8 scripts/thermo/train_alpha_guided.py \
        --schedule alpha_guided --seed 42 \
        --output /fsx/dev/jiaqi/thermo_results/alpha_guided/alpha_guided_s42.jsonl
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Training config matching Pythia-410M
CONFIG = {
    "hidden_dim": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "vocab_size": 50304,
    "seq_len": 2048,
    "batch_size_tokens": 2_097_152,  # 2M tokens per step
    "total_steps": 25_000,
    "warmup_steps": 250,
    "peak_lr": 3.0e-4,
    "min_lr": 3.0e-5,
    "weight_decay": 0.1,
    "alpha_measure_interval": 500,  # measure α every 500 steps
    "alpha_reversal_patience": 3,    # 3 consecutive increases = reversal
    "fallback_decay_start": 0.80,    # force decay at 80% if no reversal
}


def fit_alpha_fast(model, sample_layers: int = 4) -> float:
    """Fast α estimation from a few representative layers.

    Only computes SVD on `sample_layers` randomly chosen 2D layers.
    Takes ~2-5 seconds for a 410M model.
    """
    layers_2d = [(name, param) for name, param in model.named_parameters()
                 if param.ndim == 2 and min(param.shape) >= 64]

    if not layers_2d:
        return 10.0

    # Sample evenly spaced layers
    indices = np.linspace(0, len(layers_2d) - 1, sample_layers, dtype=int)
    sampled = [layers_2d[i] for i in indices]

    alphas = []
    for name, param in sampled:
        w = param.data.float()
        m, n = w.shape
        min_dim = min(m, n)

        with torch.no_grad():
            if min_dim <= 1024:
                sv = torch.linalg.svdvals(w).cpu().numpy()
            else:
                k = min(256, min_dim)
                omega = torch.randn(n, k + 16, device=w.device, dtype=torch.float32)
                y = w @ omega
                q, _ = torch.linalg.qr(y)
                b = q.T @ w
                sv = torch.linalg.svdvals(b)[:k].cpu().numpy()

        sv_pos = sv[sv > 1e-10]
        if len(sv_pos) < 10:
            continue

        eigenvalues = sv_pos ** 2
        eig_pos = eigenvalues[eigenvalues > 1e-20]
        n_eig = len(eig_pos)

        start_idx = max(1, int(n_eig * 0.02))
        end_idx = max(start_idx + 5, int(n_eig * 0.80))

        log_rank = np.log10(np.arange(start_idx, end_idx) + 1)
        log_eig = np.log10(eig_pos[start_idx:end_idx])

        if len(log_rank) < 5:
            continue

        coeffs = np.polyfit(log_rank, log_eig, 1)
        slope = coeffs[0]
        alpha = -2.0 / slope if slope < -0.01 else 20.0
        alpha = min(max(alpha, 1.0), 20.0)
        alphas.append(alpha)

    return float(np.mean(alphas)) if alphas else 10.0


def cosine_lr(step: int, cfg: dict) -> float:
    """Standard cosine LR schedule."""
    if step < cfg["warmup_steps"]:
        return cfg["peak_lr"] * step / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / (cfg["total_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["peak_lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def alpha_guided_lr(step: int, cfg: dict, decay_start_step: int = None) -> float:
    """α-guided adaptive LR schedule.

    Constant LR until α reversal detected, then linear decay.
    """
    if step < cfg["warmup_steps"]:
        return cfg["peak_lr"] * step / cfg["warmup_steps"]

    if decay_start_step is None:
        return cfg["peak_lr"]

    if step < decay_start_step:
        return cfg["peak_lr"]

    # Linear decay from reversal point to end
    remaining = cfg["total_steps"] - decay_start_step
    if remaining <= 0:
        return cfg["min_lr"]
    progress = (step - decay_start_step) / remaining
    progress = min(progress, 1.0)
    return cfg["peak_lr"] - progress * (cfg["peak_lr"] - cfg["min_lr"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True, choices=["cosine", "alpha_guided"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output JSONL for measurements")
    parser.add_argument("--model-name", default="EleutherAI/pythia-410m-deduped",
                        help="HuggingFace model to initialize from (uses step0 weights)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CONFIG

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.manual_seed(args.seed)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Load model (Pythia-410M architecture, random init or step0)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if local_rank == 0:
        print(f"Schedule: {args.schedule}, Seed: {args.seed}")
        print(f"Loading model from {args.model_name} (step0 weights)...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision="step0",
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["peak_lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    # α tracking state
    alpha_history = []
    decay_start_step = None
    measurements = []

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if local_rank == 0:
        print(f"Starting training: {cfg['total_steps']} steps")

    # Training loop (using random data as proxy — same for both schedules)
    # In a real experiment, use actual training data (The Pile)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    vocab_size = model.config.vocab_size

    for step in range(cfg["total_steps"]):
        # Determine LR
        if args.schedule == "cosine":
            lr = cosine_lr(step, cfg)
        else:
            lr = alpha_guided_lr(step, cfg, decay_start_step)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward pass with random data (proxy training)
        input_ids = torch.randint(0, vocab_size, (1, cfg["seq_len"]), device=device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        # Backward + step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Measure α periodically
        if step > 0 and step % cfg["alpha_measure_interval"] == 0 and local_rank == 0:
            alpha = fit_alpha_fast(model, sample_layers=4)
            alpha_history.append(alpha)

            # Check for reversal (only for alpha_guided schedule)
            if args.schedule == "alpha_guided" and decay_start_step is None:
                patience = cfg["alpha_reversal_patience"]
                if len(alpha_history) >= patience + 1:
                    recent = alpha_history[-(patience + 1):]
                    if all(recent[i + 1] > recent[i] for i in range(patience)):
                        decay_start_step = step
                        print(f"  ⚠ α REVERSAL detected at step {step}! Starting decay.")
                        print(f"    α history: {[f'{a:.2f}' for a in recent]}")

                # Fallback
                if step >= cfg["total_steps"] * cfg["fallback_decay_start"]:
                    decay_start_step = step
                    print(f"  ⚠ Fallback: forcing decay start at step {step} (80% reached)")

            # Record measurement
            record = {
                "step": step,
                "lr": lr,
                "loss": loss.item(),
                "alpha": alpha,
                "schedule": args.schedule,
                "seed": args.seed,
                "decay_start_step": decay_start_step,
            }
            measurements.append(record)

            if step % 2500 == 0:
                print(f"  step {step}: loss={loss.item():.4f}, α={alpha:.2f}, lr={lr:.2e}")

    # Save measurements
    if local_rank == 0:
        with open(output_path, "w") as f:
            for r in measurements:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(measurements)} measurements to {output_path}")
        print(f"Final: loss={loss.item():.4f}, α={alpha_history[-1]:.2f}")
        if decay_start_step:
            print(f"Decay started at step {decay_start_step} ({decay_start_step/cfg['total_steps']*100:.1f}%)")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
