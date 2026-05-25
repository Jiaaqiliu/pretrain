"""3-way schedule comparison at 1B scale: Cosine vs WSD vs α-Guided.

Trains Pythia-1B from step0 checkpoint on FineWeb-Edu tokenized data.
Same experimental setup as 410M but at 2.5× scale.

Usage:
    torchrun --nproc_per_node=8 scripts/thermo/train_1b_3way.py \
        --schedule cosine --seed 42 \
        --data-dir /fsx/dev/jiaqi/data/fineweb_pythia \
        --output /fsx/dev/jiaqi/thermo_results/real_3way_1b/cosine_s42.log \
        --save-dir /fsx/dev/jiaqi/thermo_results/real_3way_1b/ckpts/cosine_s42
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

CONFIG = {
    "model_name": "EleutherAI/pythia-1b-deduped",
    "hidden_dim": 2048,
    "num_layers": 16,
    "num_heads": 16,
    "vocab_size": 50304,
    "seq_len": 2048,
    "micro_batch_size": 4,
    "grad_accum_steps": 16,
    "total_steps": 9_500,
    "warmup_steps": 500,
    "peak_lr": 2.5e-4,
    "min_lr": 2.5e-5,
    "weight_decay": 0.1,
    "alpha_measure_interval": 500,
    "alpha_reversal_patience": 3,
    "fallback_decay_start": 0.80,
    "log_interval": 50,
    "save_interval": 2000,
    "wsd_stable_fraction": 0.80,
}


class TokenDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 2048):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)
        shard_files = sorted(self.data_dir.glob("shard_*.npy"))
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.npy files in {data_dir}")
        self.shards = []
        self.cumulative_len = [0]
        total_seqs = 0
        for sf in shard_files:
            mmap = np.load(sf, mmap_mode='r')
            self.shards.append(mmap)
            total_seqs += mmap.shape[0]
            self.cumulative_len.append(total_seqs)
        self.total_sequences = total_seqs
        print(f"TokenDataset: {len(shard_files)} shards, {total_seqs} sequences, "
              f"{total_seqs * seq_len / 1e9:.2f}B tokens")

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        shard_idx = 0
        for i in range(len(self.cumulative_len) - 1):
            if idx < self.cumulative_len[i + 1]:
                shard_idx = i
                break
        local_idx = idx - self.cumulative_len[shard_idx]
        tokens = self.shards[shard_idx][local_idx].astype(np.int64)
        return torch.from_numpy(tokens)


def fit_alpha_fast(model, sample_layers: int = 8) -> float:
    if hasattr(model, 'module'):
        named_params = list(model.module.named_parameters())
    else:
        named_params = list(model.named_parameters())

    layers_2d = [(name, param) for name, param in named_params
                 if param.ndim == 2 and min(param.shape) >= 64]
    if not layers_2d:
        return 10.0

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


def compute_stable_rank(model) -> float:
    if hasattr(model, 'module'):
        named_params = list(model.module.named_parameters())
    else:
        named_params = list(model.named_parameters())

    layers_2d = [(name, param) for name, param in named_params
                 if param.ndim == 2 and min(param.shape) >= 64]
    if not layers_2d:
        return 1.0

    srs = []
    for name, param in layers_2d:
        w = param.data.float()
        with torch.no_grad():
            frob_sq = (w * w).sum().item()
            v = torch.randn(w.shape[1], device=w.device, dtype=torch.float32)
            for _ in range(10):
                u = w @ v
                u = u / (u.norm() + 1e-12)
                v = w.T @ u
                v = v / (v.norm() + 1e-12)
            sigma1_sq = ((w @ v) ** 2).sum().item()
            sr = frob_sq / (sigma1_sq + 1e-12)
        srs.append(sr)

    return float(np.mean(srs)) / CONFIG["hidden_dim"]


def cosine_lr(step: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["peak_lr"] * step / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / (cfg["total_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["peak_lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def wsd_lr(step: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["peak_lr"] * step / cfg["warmup_steps"]
    stable_end = cfg["warmup_steps"] + int((cfg["total_steps"] - cfg["warmup_steps"]) * cfg["wsd_stable_fraction"])
    if step < stable_end:
        return cfg["peak_lr"]
    progress = (step - stable_end) / (cfg["total_steps"] - stable_end)
    return cfg["peak_lr"] - progress * (cfg["peak_lr"] - cfg["min_lr"])


def alpha_guided_lr(step: int, cfg: dict, decay_start_step=None) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["peak_lr"] * step / cfg["warmup_steps"]
    if decay_start_step is None:
        return cfg["peak_lr"]
    if step < decay_start_step:
        return cfg["peak_lr"]
    remaining = cfg["total_steps"] - decay_start_step
    if remaining <= 0:
        return cfg["min_lr"]
    progress = min((step - decay_start_step) / remaining, 1.0)
    return cfg["peak_lr"] - progress * (cfg["peak_lr"] - cfg["min_lr"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True, choices=["cosine", "wsd", "alpha_guided"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CONFIG.copy()
    if args.total_steps:
        cfg["total_steps"] = args.total_steps

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if local_rank == 0:
        print(f"=== 1B Scale 3-Way Experiment ===")
        print(f"Schedule: {args.schedule}, Seed: {args.seed}")
        print(f"World size: {world_size}, Device: {device}")
        print(f"Config: {cfg['total_steps']} steps, batch={cfg['micro_batch_size']}x{world_size}x{cfg['grad_accum_steps']}")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], revision="step0"
    ).to(device)

    if local_rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {cfg['model_name']} step0 ({n_params/1e6:.1f}M params)")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    dataset = TokenDataset(args.data_dir, seq_len=cfg["seq_len"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["micro_batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["peak_lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    alpha_history = []
    decay_start_step = None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(output_path, "w") if local_rank == 0 else None

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model.train()
    data_iter = iter(dataloader)
    epoch = 0
    step = 0
    running_loss = 0.0
    t_start = time.time()

    if local_rank == 0:
        tokens_per_step = cfg["micro_batch_size"] * world_size * cfg["grad_accum_steps"] * cfg["seq_len"]
        print(f"Tokens per step: {tokens_per_step:,} ({tokens_per_step/1e6:.2f}M)")
        print(f"Total tokens: {tokens_per_step * cfg['total_steps'] / 1e9:.2f}B")

    while step < cfg["total_steps"]:
        if args.schedule == "cosine":
            lr = cosine_lr(step, cfg)
        elif args.schedule == "wsd":
            lr = wsd_lr(step, cfg)
        else:
            lr = alpha_guided_lr(step, cfg, decay_start_step)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(cfg["grad_accum_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / cfg["grad_accum_steps"]
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += accum_loss
        step += 1

        if step % cfg["log_interval"] == 0 and local_rank == 0:
            avg_loss = running_loss / cfg["log_interval"]
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed
            eta_min = (cfg["total_steps"] - step) / steps_per_sec / 60
            msg = (f"  step {step}/{cfg['total_steps']}: loss={avg_loss:.4f}, "
                   f"lr={lr:.2e}, {steps_per_sec:.2f} steps/s, ETA={eta_min:.0f}min")
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
            running_loss = 0.0

        if step % cfg["alpha_measure_interval"] == 0 and local_rank == 0:
            alpha = fit_alpha_fast(model, sample_layers=8)
            sr_d = compute_stable_rank(model)
            alpha_history.append(alpha)

            if args.schedule == "alpha_guided" and decay_start_step is None:
                patience = cfg["alpha_reversal_patience"]
                if len(alpha_history) >= patience + 1:
                    recent = alpha_history[-(patience + 1):]
                    if all(recent[i + 1] > recent[i] for i in range(patience)):
                        decay_start_step = step
                        print(f"  >>> ALPHA REVERSAL at step {step}! history={[f'{a:.3f}' for a in recent]}")
                        log_file.write(f"  >>> ALPHA REVERSAL at step {step}!\n")

                if step >= int(cfg["total_steps"] * cfg["fallback_decay_start"]):
                    decay_start_step = step
                    print(f"  >>> Fallback: decay at step {step} (80%)")
                    log_file.write(f"  >>> Fallback decay at step {step}\n")

            msg = f"  [SPECTRAL] step {step}: α={alpha:.3f}, SR/d={sr_d:.4f}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        if args.save_dir and step % cfg["save_interval"] == 0 and local_rank == 0:
            ckpt_path = Path(args.save_dir) / f"step_{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            m = model.module if hasattr(model, 'module') else model
            m.save_pretrained(ckpt_path)
            print(f"  [SAVE] {ckpt_path}")

    if local_rank == 0:
        alpha_final = fit_alpha_fast(model, sample_layers=8)
        sr_d_final = compute_stable_rank(model)

        summary = (
            f"\n=== TRAINING COMPLETE ===\n"
            f"Schedule: {args.schedule}, Seed: {args.seed}\n"
            f"Final loss: {accum_loss:.4f}\n"
            f"Final alpha: {alpha_final:.3f}\n"
            f"Final SR/d: {sr_d_final:.4f}\n"
            f"Decay started at: step {decay_start_step}\n"
            f"Total time: {(time.time()-t_start)/3600:.2f} hours\n"
        )
        print(summary)
        log_file.write(summary)

        if args.save_dir:
            ckpt_path = Path(args.save_dir) / "final"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            m = model.module if hasattr(model, 'module') else model
            m.save_pretrained(ckpt_path)
            print(f"Final checkpoint: {ckpt_path}")

        log_file.close()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
