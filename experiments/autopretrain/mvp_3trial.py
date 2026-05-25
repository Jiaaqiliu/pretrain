"""MVP: 3-trial validation of data mixture hypothesis.

Trains OLMo2-1B (1.6B params) for 5000 steps with 3 different data mixes.
Compares val loss to determine if mix composition matters for reasoning.

Trials:
  1. llama3: web=52%, code=18%, math=26%, academic=4% (strongest baseline)
  2. reasoning_heavy: web=25%, code=30%, math=30%, academic=15% (our hypothesis)
  3. uniform: web=25%, code=25%, math=25%, academic=25% (control)

Each trial: 1 node × 8 H200, ~1.5 hours.
Run all 3 in parallel: ~1.5 hours total, 36 GPU-hours.

Usage (on cluster):
    torchrun --nproc-per-node=8 experiments/autopretrain/mvp_3trial.py --trial llama3
    torchrun --nproc-per-node=8 experiments/autopretrain/mvp_3trial.py --trial reasoning_heavy
    torchrun --nproc-per-node=8 experiments/autopretrain/mvp_3trial.py --trial uniform
"""

import argparse
import json
import sys
from pathlib import Path

_olmo_core_src = Path(__file__).resolve().parents[2] / "olmo-core" / "src"
if _olmo_core_src.exists():
    sys.path.insert(0, str(_olmo_core_src))

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    MetricSaverCallback,
    SpeedMonitorCallback,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerDataParallelConfig


# Data paths on FSx — raw binary format (no .npy header)
# olmo-core reads files via raw byte offset, expects headerless uint32 arrays
DOMAIN_PATHS = {
    "web": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/dclm_web/*.bin",
    "code": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/code/*.bin",
    "math": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/math/*.bin",
    "academic": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/fineweb_edu/*.bin",
}

# 3 trial mixes (must sum to 1.0)
TRIAL_MIXES = {
    "llama3": {"web": 0.52, "code": 0.18, "math": 0.26, "academic": 0.04},
    "reasoning_heavy": {"web": 0.25, "code": 0.30, "math": 0.30, "academic": 0.15},
    "uniform": {"web": 0.25, "code": 0.25, "math": 0.25, "academic": 0.25},
}

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 128 * SEQUENCE_LENGTH  # 128 sequences × 4096 = ~524K tokens/step
MAX_STEPS = 5000  # ~2.6B tokens total per trial
SAVE_FOLDER_BASE = "/fsx/dev/jiaqi/checkpoints/autopretrain-mvp"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=str, required=True, choices=list(TRIAL_MIXES.keys()))
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save-folder", type=str, default=None)
    return parser.parse_args()


def build_data_paths(mix: dict[str, float]) -> list[str]:
    """Build data paths as glob patterns for olmo-core.

    For MVP, include all domains. The actual mix ratio is approximated by
    the relative number of shards (web has 3759, code 967, math 350, academic 3759).
    Phase 1 full search will use SourceMixtureDatasetConfig for precise mixing.
    """
    paths = []
    for domain in mix.keys():
        if domain in DOMAIN_PATHS:
            paths.append(DOMAIN_PATHS[domain])
    return paths


def main():
    args = parse_args()
    mix = TRIAL_MIXES[args.trial]
    save_folder = args.save_folder or f"{SAVE_FOLDER_BASE}/{args.trial}"

    prepare_training_environment(seed=42)

    tokenizer = TokenizerConfig.dolma2()

    # Model
    model_config = TransformerConfig.olmo2_1B(vocab_size=tokenizer.padded_vocab_size())
    model = model_config.build(init_device="meta")

    # Train module
    # rank_microbatch_size = tokens processed per GPU per microbatch step
    # 4 seq × 4096 = 16384 tokens per GPU (conservative for 1.6B on H200)
    # With global_batch=128 seq and 8 GPUs: 128/8=16 seq/GPU, split into 4 microbatches
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=4 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)),
        scheduler=CosWithWarmup(warmup=500),
        max_grad_norm=1.0,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )
    train_module = train_module_config.build(model)

    # Data
    data_paths = build_data_paths(mix)
    dataset_config = NumpyFSLDatasetConfig(
        paths=data_paths,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        expand_glob=True,
    )
    loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=42,
        ignore_fingerprint_mismatch=True,
    )
    dataset = dataset_config.build()
    data_loader = loader_config.build(dataset, dp_process_group=train_module.dp_process_group)

    # Trainer
    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            max_duration=Duration.steps(args.steps),
            metrics_collect_interval=50,
        )
        .with_callback("checkpointer", CheckpointerCallback(save_interval=args.steps))
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("gc", GarbageCollectorCallback())
        .with_callback("metric_saver", MetricSaverCallback())
    )

    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    teardown_training_environment()


if __name__ == "__main__":
    main()
