"""Causality test: fork training at α reversal point with decay vs no-decay.

Takes a saved checkpoint (from the α-guided run's decay trigger point) and
continues training with two different strategies:
  - Branch A (obey): Apply linear LR decay from peak to min
  - Branch B (ignore): Continue at peak LR, then abrupt drop in last 5%

This demonstrates the CAUSAL value of the α reversal signal.

Usage:
    # Branch A: Obey the α signal (decay LR)
    torchrun --nproc_per_node=8 scripts/thermo/train_causality_fork.py \
        --branch obey \
        --checkpoint /path/to/alpha_guided/step_XXXX \
        --data-dir /fsx/dev/jiaqi/data/fineweb_pythia \
        --remaining-steps 2000 \
        --output /fsx/dev/jiaqi/thermo_results/causality/obey.log \
        --save-dir /fsx/dev/jiaqi/thermo_results/causality/obey_ckpt

    # Branch B: Ignore the α signal (continue peak LR)
    torchrun --nproc_per_node=8 scripts/thermo/train_causality_fork.py \
        --branch ignore \
        --checkpoint /path/to/alpha_guided/step_XXXX \
        --data-dir /fsx/dev/jiaqi/data/fineweb_pythia \
        --remaining-steps 2000 \
        --output /fsx/dev/jiaqi/thermo_results/causality/ignore.log \
        --save-dir /fsx/dev/jiaqi/thermo_results/causality/ignore_ckpt
"""

import argparse
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
    "hidden_dim": 2048,
    "seq_len": 2048,
    "micro_batch_size": 4,
    "grad_accum_steps": 16,
    "peak_lr": 2.5e-4,
    "min_lr": 2.5e-5,
    "weight_decay": 0.1,
    "alpha_measure_interval": 200,
    "log_interval": 50,
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
    layers_2d = [(n, p) for n, p in named_params if p.ndim == 2 and min(p.shape) >= 64]
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


def compute_stable_rank(model, hidden_dim) -> float:
    if hasattr(model, 'module'):
        named_params = list(model.module.named_parameters())
    else:
        named_params = list(model.named_parameters())
    layers_2d = [(n, p) for n, p in named_params if p.ndim == 2 and min(p.shape) >= 64]
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
    return float(np.mean(srs)) / hidden_dim


def get_lr(step, total_steps, branch, cfg):
    if branch == "obey":
        progress = step / total_steps
        return cfg["peak_lr"] - progress * (cfg["peak_lr"] - cfg["min_lr"])
    else:
        if step < int(total_steps * 0.95):
            return cfg["peak_lr"]
        progress = (step - int(total_steps * 0.95)) / (total_steps * 0.05)
        return cfg["peak_lr"] - progress * (cfg["peak_lr"] - cfg["min_lr"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True, choices=["obey", "ignore"])
    parser.add_argument("--checkpoint", required=True, help="Path to fork checkpoint")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--remaining-steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CONFIG.copy()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if local_rank == 0:
        print(f"=== Causality Fork: Branch={args.branch.upper()} ===")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Remaining steps: {args.remaining_steps}")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)

    if local_rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params/1e6:.1f}M params")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    dataset = TokenDataset(args.data_dir, seq_len=cfg["seq_len"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=cfg["micro_batch_size"], sampler=sampler,
                           num_workers=4, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["peak_lr"],
                                   weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))

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
    total_steps = args.remaining_steps

    while step < total_steps:
        lr = get_lr(step, total_steps, args.branch, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(cfg["grad_accum_steps"]):
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
            msg = f"  step {step}/{total_steps}: loss={avg_loss:.4f}, lr={lr:.2e}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
            running_loss = 0.0

        if step % cfg["alpha_measure_interval"] == 0 and local_rank == 0:
            alpha = fit_alpha_fast(model)
            sr_d = compute_stable_rank(model, cfg["hidden_dim"])
            msg = f"  [SPECTRAL] step {step}: α={alpha:.3f}, SR/d={sr_d:.4f}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

    if local_rank == 0:
        alpha_final = fit_alpha_fast(model)
        sr_d_final = compute_stable_rank(model, cfg["hidden_dim"])
        summary = (
            f"\n=== FORK COMPLETE: {args.branch.upper()} ===\n"
            f"Final loss: {accum_loss:.4f}\n"
            f"Final α: {alpha_final:.3f}\n"
            f"Final SR/d: {sr_d_final:.4f}\n"
            f"Time: {(time.time()-t_start)/3600:.2f}h\n"
        )
        print(summary)
        log_file.write(summary)
        log_file.close()

        if args.save_dir:
            ckpt_path = Path(args.save_dir) / "final"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            m = model.module if hasattr(model, 'module') else model
            m.save_pretrained(ckpt_path)
            print(f"  Saved: {ckpt_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
