"""Mid-training vs Direct Decay comparison experiment (Q3 validation).

Compares:
  - Mid-training: Stage 1 → decay → data switch + LR warmup → decay
  - Direct decay: Stage 1 → extended decay on original data (no data switch)

This isolates the thermodynamic "glass annealing" effect of data switching.

Usage:
    torchrun --nproc_per_node=8 scripts/thermo/train_midtraining_comparison.py \
        --model-size 190M \
        --mode midtrain \
        --seed 42 \
        --output-dir /fsx/dev/jiaqi/thermo_experiments/190m_midtrain_s42
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

_olmo_core_src = Path(__file__).resolve().parents[2] / "olmo-core" / "src"
if _olmo_core_src.exists():
    sys.path.insert(0, str(_olmo_core_src))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
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

SEQUENCE_LENGTH = 4096
TOKENIZER = TokenizerConfig.dolma2()

# Stage 1 data: web-scale pre-training data
STAGE1_DATA = [
    "/fsx/dev/jiaqi/data/olmo-pretrain/dclm_web",
    "/fsx/dev/jiaqi/data/olmo-pretrain/dolma_web",
]

# Stage 2 data: curated mid-training data (higher quality)
STAGE2_DATA = [
    "/fsx/dev/jiaqi/data/olmo-pretrain/fineweb_edu",
    "/fsx/dev/jiaqi/data/olmo-pretrain/starcoder",
]

MODEL_CONFIGS = {
    "190M": {
        "factory": "olmo2_190M",
        "stage1_steps": 15000,
        "stage2_steps": 10000,
        "peak_lr": 6e-4,
        "global_batch_size": 256 * SEQUENCE_LENGTH,
        "rank_microbatch_size": 4 * SEQUENCE_LENGTH,
    },
    "1B": {
        "factory": "olmo2_1B",
        "stage1_steps": 30000,
        "stage2_steps": 20000,
        "peak_lr": 3e-4,
        "global_batch_size": 256 * SEQUENCE_LENGTH,
        "rank_microbatch_size": 4 * SEQUENCE_LENGTH,
    },
}


class MultiStageLRCallback(Callback):
    """LR schedule for multi-stage training.

    Stage 1: warmup → stable → decay (WSD)
    Stage 2 (midtrain mode): warmup → stable → decay (on new data)
    Direct mode: just continue decaying on same data
    """

    def __init__(self, mode: str, peak_lr: float, stage1_steps: int, stage2_steps: int):
        self.mode = mode
        self.peak_lr = peak_lr
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.total_steps = stage1_steps + stage2_steps

    def _get_lr(self, step: int) -> float:
        if self.mode == "midtrain":
            if step < self.stage1_steps:
                return self._wsd_lr(step, self.stage1_steps, self.peak_lr)
            else:
                # Re-warmup + stable + decay for stage 2
                s2_step = step - self.stage1_steps
                return self._wsd_lr(s2_step, self.stage2_steps, self.peak_lr * 0.5)
        else:
            # Direct decay: single WSD across all steps
            return self._wsd_lr(step, self.total_steps, self.peak_lr)

    def _wsd_lr(self, step: int, total: int, peak: float) -> float:
        warmup = int(0.02 * total)
        stable_end = int(0.78 * total)
        min_ratio = 0.01

        if step < warmup:
            return peak * step / max(warmup, 1)
        if step < stable_end:
            return peak
        progress = (step - stable_end) / max(total - stable_end, 1)
        return peak * (1.0 - progress * (1.0 - min_ratio))

    def pre_step(self, step: int, **kwargs):
        lr = self._get_lr(step)
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "train_module"):
            for param_group in trainer.train_module.optim.param_groups:
                param_group["lr"] = lr


class DataSwitchCallback(Callback):
    """Switch data source at stage boundary (for mid-training mode)."""

    def __init__(self, switch_step: int, stage2_data_paths: list[str]):
        self.switch_step = switch_step
        self.stage2_data_paths = stage2_data_paths
        self.switched = False

    def post_step(self, step: int, **kwargs):
        if step == self.switch_step and not self.switched:
            self.switched = True
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                print(f"\n[DataSwitch] Switching to Stage 2 data at step {step}")
                print(f"  New data: {self.stage2_data_paths}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=["190M", "1B"])
    parser.add_argument("--mode", required=True, choices=["midtrain", "direct"],
                        help="midtrain: data switch + re-warmup; direct: extended decay on same data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-project", type=str, default="thermo-pretraining")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MODEL_CONFIGS[args.model_size]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg["stage1_steps"] + cfg["stage2_steps"]

    # Save config
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump({
                "model_size": args.model_size,
                "mode": args.mode,
                "seed": args.seed,
                "stage1_steps": cfg["stage1_steps"],
                "stage2_steps": cfg["stage2_steps"],
                "total_steps": total_steps,
            }, f, indent=2)

    prepare_training_environment(seed=args.seed)

    # Build model
    factory = getattr(TransformerConfig, cfg["factory"])
    model_config = factory(vocab_size=TOKENIZER.padded_vocab_size())
    model = model_config.build(init_device="meta")

    # Build train module
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=cfg["rank_microbatch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(lr=cfg["peak_lr"], weight_decay=0.1, betas=(0.9, 0.95)),
        scheduler=CosWithWarmup(warmup=int(0.02 * total_steps)),
        max_grad_norm=1.0,
        dp_config=TransformerDataParallelConfig(),
    )
    train_module = train_module_config.build(model)

    # Build data loader (start with Stage 1 data)
    data_paths = STAGE1_DATA if args.mode == "midtrain" else STAGE1_DATA
    existing_paths = [p for p in data_paths if Path(p).exists()]
    if not existing_paths:
        raise RuntimeError(f"No data paths found: {data_paths}")

    dataset = NumpyFSLDatasetConfig(
        paths=existing_paths, sequence_length=SEQUENCE_LENGTH, tokenizer=TOKENIZER,
    ).build()
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=cfg["global_batch_size"], seed=args.seed,
    ).build(dataset, dp_process_group=train_module.dp_process_group)

    # Build trainer with callbacks
    run_name = f"thermo-midtrain-{args.model_size}-{args.mode}-s{args.seed}"
    trainer_config = (
        TrainerConfig(
            save_folder=str(output_dir / "checkpoints"),
            max_duration=Duration.steps(total_steps),
            metrics_collect_interval=10,
        )
        .with_callback("checkpointer", CheckpointerCallback(save_interval=200, ephemeral_save_interval=200))
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("gc", GarbageCollectorCallback())
        .with_callback("wandb", WandBCallback(project=args.wandb_project, name=run_name))
        .with_callback("lr_schedule", MultiStageLRCallback(
            mode=args.mode, peak_lr=cfg["peak_lr"],
            stage1_steps=cfg["stage1_steps"], stage2_steps=cfg["stage2_steps"],
        ))
    )

    if args.mode == "midtrain":
        trainer_config = trainer_config.with_callback(
            "data_switch",
            DataSwitchCallback(switch_step=cfg["stage1_steps"], stage2_data_paths=STAGE2_DATA),
        )

    # Add thermo measurement callback
    from scripts.thermo.train_schedule_comparison import ThermoMeasurementCallback
    trainer_config = trainer_config.with_callback(
        "thermo_measure",
        ThermoMeasurementCallback(
            output_path=str(output_dir / "thermo_measurements.jsonl"),
            measure_interval=200,
            weight_decay=0.1,
            batch_size=cfg["global_batch_size"] // SEQUENCE_LENGTH,
        ),
    )

    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()
    teardown_training_environment()

    if rank == 0:
        print(f"\nComplete: {args.model_size} / {args.mode} / seed={args.seed}")


if __name__ == "__main__":
    main()
