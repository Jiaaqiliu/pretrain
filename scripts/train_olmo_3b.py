"""OLMo-3B Pre-training Script.

使用 OLMo-core 训练引擎在 H200 集群上预训练 3B 参数的语言模型。
配合 PyTorchJob 通过 torchrun 启动。

架构: OLMo2-3B (d_model=3328, n_layers=16, n_heads=16, ~3B params)
数据: FineWeb-Edu tokenized numpy (dolma2 tokenizer, vocab_size=100278)
优化器: AdamW (lr=3e-4, wd=0.1) + CosWithWarmup (2000 steps)
批次: 256 sequences × 4096 tokens = ~1M tokens/step
总训练: 60000 steps × 1M tokens/step = ~60B tokens

用法 (单机测试):
    torchrun --nproc-per-node=8 scripts/train_olmo_3b.py

用法 (多机，由 PyTorchJob 自动设置环境变量):
    torchrun --nproc_per_node=8 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        scripts/train_olmo_3b.py
"""

import sys
from pathlib import Path

_olmo_core_src = Path(__file__).resolve().parent.parent / "olmo-core" / "src"
if _olmo_core_src.exists():
    sys.path.insert(0, str(_olmo_core_src))

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerDataParallelConfig

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 256 * SEQUENCE_LENGTH  # ~1M tokens/step
MAX_STEPS = 60_000  # ~60B tokens total (Chinchilla optimal for 3B)
SAVE_FOLDER = "/fsx/dev/jiaqi/checkpoints/olmo-3b-pretrain"

DATA_PATHS = [
    "/fsx/dev/jiaqi/data/olmo-3b-pretrain/web",
    "/fsx/dev/jiaqi/data/olmo-3b-pretrain/code",
    "/fsx/dev/jiaqi/data/olmo-3b-pretrain/math",
    "/fsx/dev/jiaqi/data/olmo-3b-pretrain/books",
    "/fsx/dev/jiaqi/data/olmo-3b-pretrain/academic",
]

TOKENIZER = TokenizerConfig.dolma2()


def build_model_config() -> TransformerConfig:
    return TransformerConfig.olmo2_3B(vocab_size=TOKENIZER.padded_vocab_size())


def build_train_module_config() -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=4 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        ),
        scheduler=CosWithWarmup(warmup=2000),
        max_grad_norm=1.0,
        dp_config=TransformerDataParallelConfig(),
    )


def build_data_loader(train_module):
    existing_paths = [p for p in DATA_PATHS if Path(p).exists()]
    if not existing_paths:
        raise RuntimeError(
            f"No data paths found. Expected tokenized data at: {DATA_PATHS}. "
            f"Run scripts/prepare_data_3b.py first."
        )

    dataset_config = NumpyFSLDatasetConfig(
        paths=existing_paths,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=TOKENIZER,
    )
    loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=42,
    )
    dataset = dataset_config.build()
    return loader_config.build(dataset, dp_process_group=train_module.dp_process_group)


def build_trainer_config() -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=SAVE_FOLDER,
            max_duration=Duration.steps(MAX_STEPS),
            metrics_collect_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=2000, ephemeral_save_interval=500),
        )
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("gc", GarbageCollectorCallback())
        .with_callback(
            "wandb",
            WandBCallback(project="a-evolve-olmo-3b", name="olmo-3b-pretrain"),
        )
    )


def main():
    prepare_training_environment(seed=42)

    model_config = build_model_config()
    model = model_config.build(init_device="meta")

    train_module_config = build_train_module_config()
    train_module = train_module_config.build(model)

    data_loader = build_data_loader(train_module)

    trainer_config = build_trainer_config()
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    teardown_training_environment()


if __name__ == "__main__":
    main()
