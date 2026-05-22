"""Proxy training experiment: compare LR schedules with thermodynamic instrumentation.

Trains OLMo-190M or OLMo-1B from scratch with one of four schedules:
  - cosine: standard cosine annealing
  - wsd_linear: warmup-stable-decay (linear)
  - wsd_exponential: warmup-stable-decay (exponential)
  - gaussian: our thermodynamically-derived Gaussian decay

Includes dense checkpointing (every 200 steps) and thermodynamic measurement
callback for fine-grained tracking.

Usage (single node, 8 GPUs):
    torchrun --nproc_per_node=8 scripts/thermo/train_schedule_comparison.py \
        --model-size 190M \
        --schedule gaussian \
        --seed 42 \
        --output-dir /fsx/dev/jiaqi/thermo_experiments/190m_gaussian_s42

Usage (multi-node, via PyTorchJob):
    torchrun --nproc_per_node=8 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        scripts/thermo/train_schedule_comparison.py \
        --model-size 1B --schedule gaussian --seed 42 \
        --output-dir /fsx/dev/jiaqi/thermo_experiments/1b_gaussian_s42
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

_olmo_core_src = Path(__file__).resolve().parents[2] / "olmo-core" / "src"
if _olmo_core_src.exists():
    sys.path.insert(0, str(_olmo_core_src))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.composable import (
    NumpyDocumentSource,
    ConcatAndChunkInstanceSource,
    ComposableDataLoaderConfig,
)
import numpy as np
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerDataParallelConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.config import DType

from experiments.thermodynamics.schedules import get_schedule

# =============================================================================
# Model configurations
# =============================================================================

MODEL_CONFIGS = {
    "190M": {
        "factory": "olmo2_190M",
        "max_steps": 25_000,   # ~25B tokens (matches measurement window)
        "peak_lr": 6e-4,
        "global_batch_size": 256 * 4096,  # ~1M tokens/step
        "rank_microbatch_size": 4 * 4096,
        "nodes": 1,
    },
    "1B": {
        "factory": "olmo2_1B",
        "max_steps": 50_000,   # ~50B tokens
        "peak_lr": 3e-4,
        "global_batch_size": 256 * 4096,
        "rank_microbatch_size": 4 * 4096,
        "nodes": 2,
    },
}

SEQUENCE_LENGTH = 4096
TOKENIZER = TokenizerConfig(
    identifier="allenai/OLMo-2-0325-32B",
    vocab_size=100278,
    eos_token_id=100257,
    pad_token_id=100277,
)

DATA_PATHS_190M = [
    "/fsx/dev/jiaqi/data/olmo-pretrain/dclm_web",
    "/fsx/dev/jiaqi/data/olmo-pretrain/dolma_web",
    "/fsx/dev/jiaqi/data/olmo-pretrain/code",
]

DATA_PATHS_1B = [
    "/fsx/dev/jiaqi/data/olmo-pretrain/dclm_web",
    "/fsx/dev/jiaqi/data/olmo-pretrain/dolma_web",
    "/fsx/dev/jiaqi/data/olmo-pretrain/code",
    "/fsx/dev/jiaqi/data/olmo-pretrain/math",
    "/fsx/dev/jiaqi/data/olmo-pretrain/books",
]


# =============================================================================
# Custom LR Schedule Callback
# =============================================================================


class LRScheduleCallback(Callback):
    """Override OLMo-core's built-in scheduler with our custom schedule.

    Manually sets the learning rate at each step according to the chosen schedule.
    """

    def __init__(self, schedule_name: str, total_steps: int, peak_lr: float):
        self.schedule_fn = get_schedule(schedule_name)
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.schedule_name = schedule_name

    def pre_step(self, step: int, **kwargs):
        lr = self.schedule_fn(step, self.total_steps, self.peak_lr)
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "train_module"):
            for param_group in trainer.train_module.optim.param_groups:
                param_group["lr"] = lr


# =============================================================================
# Thermodynamic Measurement Callback
# =============================================================================


class ThermoMeasurementCallback(Callback):
    """Compute and log thermodynamic state variables during training.

    Runs every `measure_interval` steps. Logs to a JSONL file and optionally WandB.
    """

    def __init__(
        self,
        output_path: str,
        measure_interval: int = 200,
        svd_k: int = 256,
        weight_decay: float = 0.1,
        batch_size: int = 256,
    ):
        self.output_path = output_path
        self.measure_interval = measure_interval
        self.svd_k = svd_k
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self._file = None

    def post_step(self, step: int, **kwargs):
        if step % self.measure_interval != 0:
            return

        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        # Only measure on rank 0
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank != 0:
            return

        from experiments.thermodynamics.measures import (
            global_spectral_entropy,
            weight_volume,
            order_parameter_layer,
        )

        t0 = time.time()

        # Get current LR
        lr = 0.0
        for pg in trainer.train_module.optim.param_groups:
            lr = pg["lr"]
            break

        # Get loss from trainer metrics
        loss = 0.0
        if hasattr(trainer, "metrics") and "train/loss" in trainer.metrics:
            loss = float(trainer.metrics["train/loss"])

        # FSDP: must summon full params for correct measurements
        model = trainer.train_module.model
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            ctx = FSDP.summon_full_params(model, writeback=False, recurse=True)
        except (ImportError, TypeError, AttributeError):
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            vol = weight_volume(model)
            s_global, layer_info = global_spectral_entropy(model, k=self.svd_k)
            psi = sum(lm.order_parameter for lm in layer_info) / max(len(layer_info), 1)
            n_params = sum(p.numel() for p in model.parameters())

        # Gradient variance estimate (from optimizer state if available)
        grad_var = self._estimate_grad_variance_from_optimizer(trainer.train_module.optim)

        # Temperature and free energy
        temp = (lr * grad_var) / (2 * self.batch_size) if self.batch_size > 0 else 0.0
        free_e = loss - temp * s_global

        # P·V / (N·T)
        pv_over_nt = (self.weight_decay * vol) / (n_params * temp) if temp > 1e-15 else 0.0

        elapsed = time.time() - t0

        record = {
            "step": step,
            "lr": lr,
            "loss": loss,
            "volume": vol,
            "spectral_entropy": s_global,
            "order_parameter": psi,
            "temperature": temp,
            "pressure": self.weight_decay,
            "free_energy": free_e,
            "pv_over_nt": pv_over_nt,
            "grad_variance": grad_var,
            "n_params": n_params,
            "measure_time_s": elapsed,
        }

        # Write to file
        if self._file is None:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.output_path, "a")

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        print(f"[Thermo] step={step} S={s_global:.4f} ψ={psi:.4f} "
              f"F={free_e:.4f} PV/NT={pv_over_nt:.4f} ({elapsed:.1f}s)")

    def _estimate_grad_variance_from_optimizer(self, optimizer) -> float:
        """Estimate gradient variance from AdamW's second moment estimates."""
        total_var = 0.0
        total_count = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state is None:
                    continue
                if "exp_avg_sq" in state:
                    # AdamW stores E[g²], variance ≈ E[g²] - E[g]²
                    v = state["exp_avg_sq"]
                    total_var += v.sum().item()
                    total_count += v.numel()
        if total_count == 0:
            return 1e-6
        return total_var / total_count

    def __del__(self):
        if self._file is not None:
            self._file.close()


# =============================================================================
# Gradient Noise Estimation Callback
# =============================================================================


class GradNoiseCallback(Callback):
    """Estimate gradient variance using double-batch method every N steps.

    Stores results in a JSONL file for temperature computation.
    """

    def __init__(self, output_path: str, estimate_interval: int = 1000):
        self.output_path = output_path
        self.estimate_interval = estimate_interval

    def post_step(self, step: int, **kwargs):
        if step % self.estimate_interval != 0 or step == 0:
            return

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank != 0:
            return

        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        # Use optimizer's exp_avg_sq as a proxy for gradient variance
        optimizer = trainer.train_module.optim
        grad_var = 0.0
        count = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state and "exp_avg_sq" in state:
                    grad_var += state["exp_avg_sq"].mean().item()
                    count += 1

        if count > 0:
            grad_var /= count

        record = {"step": step, "grad_variance": grad_var}
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# =============================================================================
# Training Script
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Proxy training with LR schedule comparison")
    parser.add_argument("--model-size", required=True, choices=["190M", "1B"])
    parser.add_argument("--schedule", required=True, choices=["cosine", "wsd_linear", "wsd_exponential", "gaussian"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--max-steps", type=int, default=None, help="Override default max_steps")
    parser.add_argument("--measure-interval", type=int, default=200,
                        help="Steps between thermodynamic measurements")
    parser.add_argument("--checkpoint-interval", type=int, default=200,
                        help="Steps between checkpoints (dense for thermo tracking)")
    parser.add_argument("--wandb-project", type=str, default="thermo-pretraining")
    return parser.parse_args()


def build_model(model_size: str) -> TransformerConfig:
    config = MODEL_CONFIGS[model_size]
    factory = getattr(TransformerConfig, config["factory"])
    return factory(vocab_size=TOKENIZER.vocab_size)


def build_train_module(model_size: str, schedule: str) -> TransformerTrainModuleConfig:
    config = MODEL_CONFIGS[model_size]

    # Use CosWithWarmup as placeholder — will be overridden by LRScheduleCallback
    # (except for cosine schedule, where it's the actual schedule)
    scheduler = CosWithWarmup(warmup=int(0.02 * config["max_steps"]))

    return TransformerTrainModuleConfig(
        rank_microbatch_size=config["rank_microbatch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=config["peak_lr"],
            weight_decay=0.1,
            betas=(0.9, 0.95),
        ),
        scheduler=scheduler,
        max_grad_norm=1.0,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp, param_dtype=DType.bfloat16),
        autocast_precision=DType.bfloat16,
    )


def build_data_loader(model_size: str, train_module):
    import tempfile

    data_paths = DATA_PATHS_190M if model_size == "190M" else DATA_PATHS_1B
    existing_paths = [p for p in data_paths if Path(p).exists()]
    if not existing_paths:
        raise RuntimeError(f"No data paths found. Expected: {data_paths}")

    config = MODEL_CONFIGS[model_size]
    work_dir = Path(tempfile.mkdtemp(prefix="olmo_data_"))

    all_npy_files = []
    for data_dir in existing_paths:
        npy_files = sorted(Path(data_dir).glob("*.npy"))
        all_npy_files.extend([str(f) for f in npy_files])

    if not all_npy_files:
        raise RuntimeError(f"No .npy files found in: {existing_paths}")

    token_source = NumpyDocumentSource(
        source_paths=all_npy_files,
        dtype=np.uint32,
        work_dir=work_dir,
        tokenizer=TOKENIZER,
    )
    instance_source = ConcatAndChunkInstanceSource(
        token_source, sequence_length=SEQUENCE_LENGTH, work_dir=work_dir,
    )
    loader_config = ComposableDataLoaderConfig(
        global_batch_size=config["global_batch_size"],
        seed=42,
        tokenizer=TOKENIZER,
        display_source_visualization=False,
    )
    return loader_config.build(
        instance_source,
        work_dir=work_dir,
        dp_process_group=train_module.dp_process_group,
    )


def build_trainer(
    args,
    model_size: str,
) -> TrainerConfig:
    config = MODEL_CONFIGS[model_size]
    max_steps = args.max_steps or config["max_steps"]
    output_dir = Path(args.output_dir)

    run_name = f"thermo-{model_size}-{args.schedule}-s{args.seed}"

    trainer_config = (
        TrainerConfig(
            save_folder=str(output_dir / "checkpoints"),
            max_duration=Duration.steps(max_steps),
            metrics_collect_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=args.checkpoint_interval,
            ),
        )
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("gc", GarbageCollectorCallback())
    )

    # Only add WandB callback if WANDB_API_KEY is available
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "disabled":
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(project=args.wandb_project, name=run_name),
        )

    # Custom LR schedule callback (overrides built-in scheduler for non-cosine)
    if args.schedule != "cosine":
        trainer_config = trainer_config.with_callback(
            "lr_schedule",
            LRScheduleCallback(
                schedule_name=args.schedule,
                total_steps=max_steps,
                peak_lr=config["peak_lr"],
            ),
        )

    # Thermodynamic measurement callback
    trainer_config = trainer_config.with_callback(
        "thermo_measure",
        ThermoMeasurementCallback(
            output_path=str(output_dir / "thermo_measurements.jsonl"),
            measure_interval=args.measure_interval,
            weight_decay=0.1,
            batch_size=config["global_batch_size"] // SEQUENCE_LENGTH,
        ),
    )

    # Gradient noise estimation callback
    trainer_config = trainer_config.with_callback(
        "grad_noise",
        GradNoiseCallback(
            output_path=str(output_dir / "grad_noise.jsonl"),
            estimate_interval=1000,
        ),
    )

    return trainer_config


def main():
    args = parse_args()

    # Save experiment config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_config = {
        "model_size": args.model_size,
        "schedule": args.schedule,
        "seed": args.seed,
        "max_steps": args.max_steps or MODEL_CONFIGS[args.model_size]["max_steps"],
        "peak_lr": MODEL_CONFIGS[args.model_size]["peak_lr"],
        "measure_interval": args.measure_interval,
        "checkpoint_interval": args.checkpoint_interval,
    }

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump(exp_config, f, indent=2)

    prepare_training_environment(seed=args.seed)

    model_config = build_model(args.model_size)
    model = model_config.build(init_device="meta")

    train_module_config = build_train_module(args.model_size, args.schedule)
    train_module = train_module_config.build(model)

    data_loader = build_data_loader(args.model_size, train_module)

    trainer_config = build_trainer(args, args.model_size)
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    teardown_training_environment()

    if rank == 0:
        print(f"\nExperiment complete: {args.model_size} / {args.schedule} / seed={args.seed}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
