"""WSD stable-phase ablation experiment (Appendix E).

Tests prediction that WSD advantage disappears when stable phase is too short.
Trains 190M models with stable_frac ∈ {0.2, 0.4, 0.6, 0.8, 0.95}.

Usage:
    torchrun --nproc_per_node=8 scripts/thermo/train_wsd_ablation.py \
        --stable-frac 0.6 \
        --seed 42 \
        --output-dir /fsx/dev/jiaqi/thermo_experiments/ablation_stable_0.6_s42
"""

import argparse
import json
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
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.config import DType

from experiments.thermodynamics.schedules import wsd_linear_lr

SEQUENCE_LENGTH = 4096
TOKENIZER = TokenizerConfig.dolma2()
MAX_STEPS = 25000
PEAK_LR = 6e-4
GLOBAL_BATCH_SIZE = 256 * SEQUENCE_LENGTH

DATA_PATHS = [
    "/fsx/dev/jiaqi/data/olmo-pretrain/dclm_web",
    "/fsx/dev/jiaqi/data/olmo-pretrain/dolma_web",
    "/fsx/dev/jiaqi/data/olmo-pretrain/code",
]


class AblationLRCallback(Callback):
    """WSD schedule with configurable stable phase fraction."""

    def __init__(self, stable_frac: float, peak_lr: float, total_steps: int):
        self.stable_frac = stable_frac
        self.peak_lr = peak_lr
        self.total_steps = total_steps

    def pre_step(self, batch):
        step = self.trainer.global_step
        lr = wsd_linear_lr(
            step, self.total_steps, self.peak_lr,
            warmup_frac=0.02, stable_frac=self.stable_frac,
        )
        for pg in self.trainer.train_module.optim.param_groups:
            pg["lr"] = lr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-frac", type=float, required=True,
                        help="Stable phase fraction (0.2, 0.4, 0.6, 0.8, 0.95)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-project", type=str, default="thermo-pretraining")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump({
                "experiment": "wsd_ablation",
                "stable_frac": args.stable_frac,
                "seed": args.seed,
                "max_steps": MAX_STEPS,
                "peak_lr": PEAK_LR,
            }, f, indent=2)

    prepare_training_environment(seed=args.seed)

    model_config = TransformerConfig.olmo2_190M(vocab_size=TOKENIZER.padded_vocab_size())
    model = model_config.build(init_device="meta")

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=4 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(lr=PEAK_LR, weight_decay=0.1, betas=(0.9, 0.95)),
        scheduler=CosWithWarmup(warmup=int(0.02 * MAX_STEPS)),
        max_grad_norm=1.0,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp, param_dtype=DType.bfloat16),
        autocast_precision=DType.bfloat16,
    )
    train_module = train_module_config.build(model)

    existing_paths = [p for p in DATA_PATHS if Path(p).exists()]
    if not existing_paths:
        raise RuntimeError(f"No data found at {DATA_PATHS}")

    dataset = NumpyFSLDatasetConfig(
        paths=existing_paths, sequence_length=SEQUENCE_LENGTH, tokenizer=TOKENIZER,
    ).build()
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=args.seed,
    ).build(dataset, dp_process_group=train_module.dp_process_group)

    run_name = f"thermo-ablation-stable{args.stable_frac}-s{args.seed}"
    trainer_config = (
        TrainerConfig(
            save_folder=str(output_dir / "checkpoints"),
            max_duration=Duration.steps(MAX_STEPS),
            metrics_collect_interval=10,
        )
        .with_callback("checkpointer", CheckpointerCallback(save_interval=500))
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("gc", GarbageCollectorCallback())
        .with_callback("wandb", WandBCallback(project=args.wandb_project, name=run_name))
        .with_callback("lr_schedule", AblationLRCallback(
            stable_frac=args.stable_frac, peak_lr=PEAK_LR, total_steps=MAX_STEPS,
        ))
    )

    # Thermo measurement (inline import to avoid scripts/__init__.py dependency)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_schedule_comparison import ThermoMeasurementCallback
    trainer_config = trainer_config.with_callback(
        "thermo_measure",
        ThermoMeasurementCallback(
            output_path=str(output_dir / "thermo_measurements.jsonl"),
            measure_interval=500,
            weight_decay=0.1,
            batch_size=GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH,
        ),
    )

    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()
    teardown_training_environment()

    if rank == 0:
        print(f"\nAblation complete: stable_frac={args.stable_frac}, seed={args.seed}")


if __name__ == "__main__":
    main()
