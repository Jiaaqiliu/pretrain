"""OLMo-3B Pre-training Script.

使用 OLMo-core 训练引擎在 H200 集群上预训练 3B 参数的语言模型。
配合 PyTorchJob 通过 torchrun 启动。

用法 (单机测试):
    torchrun --nproc-per-node=8 scripts/train_olmo_3b.py

用法 (多机，由 PyTorchJob 自动设置环境变量):
    torchrun --nproc_per_node=8 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        scripts/train_olmo_3b.py
"""

import sys
from pathlib import Path

# 确保 olmo-core 在 Python path 中
_olmo_core_src = Path(__file__).resolve().parent.parent / "olmo-core" / "src"
if _olmo_core_src.exists():
    sys.path.insert(0, str(_olmo_core_src))

from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.common import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.data import NumpyFSLDatasetConfig, NumpyDataLoaderConfig


# ═══════════════════════════════════════════════════════
# 模型配置：OLMo-3B
# ═══════════════════════════════════════════════════════

def build_model_config() -> TransformerConfig:
    """3B 参数 Transformer 架构。"""
    return TransformerConfig(
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=100278,
        max_sequence_length=4096,
    )


# ═══════════════════════════════════════════════════════
# 训练模块配置：优化器 + 并行策略
# ═══════════════════════════════════════════════════════

def build_train_module_config() -> TransformerTrainModuleConfig:
    """训练模块：AdamW + CosineWarmup + FSDP。"""
    return TransformerTrainModuleConfig(
        # Micro-batch: 每个 rank 每步处理 4 sequences × 4096 tokens
        rank_microbatch_size=4 * 4096,
        max_sequence_length=4096,
        # 优化器
        optim=AdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        ),
        # 学习率调度
        scheduler=CosWithWarmup(
            warmup_steps=2000,
        ),
        max_grad_norm=1.0,
        # FSDP 数据并行（3B 模型 16 GPU 够了，不需要 TP）
        dp_config=TransformerDataParallelConfig(),
    )


# ═══════════════════════════════════════════════════════
# 数据配置
# ═══════════════════════════════════════════════════════

def build_data_loader(train_module):
    """从 FSx 上的 tokenized numpy 数据加载训练数据。"""
    data_paths = [
        "/fsx/shared/data/tokenized/dolma/web",
        "/fsx/shared/data/tokenized/dolma/code",
        "/fsx/shared/data/tokenized/dolma/math",
        "/fsx/shared/data/tokenized/dolma/books",
        "/fsx/shared/data/tokenized/dolma/academic",
    ]

    # 过滤存在的路径
    existing_paths = [p for p in data_paths if Path(p).exists()]
    if not existing_paths:
        raise RuntimeError(
            f"No data paths found. Expected tokenized data at: {data_paths}. "
            f"Please run data preparation first."
        )

    dataset_config = NumpyFSLDatasetConfig(
        paths=existing_paths,
        sequence_length=4096,
    )
    # Global batch size: 256 sequences × 4096 tokens = ~1M tokens/step
    loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * 4096,
        seed=42,
    )
    dataset = dataset_config.build()
    return loader_config.build(
        dataset,
        dp_process_group=train_module.dp_process_group,
    )


# ═══════════════════════════════════════════════════════
# Trainer 配置：checkpoint + 监控
# ═══════════════════════════════════════════════════════

def build_trainer_config() -> TrainerConfig:
    """Trainer: 60000 步，每 2000 步存 checkpoint。"""
    return (
        TrainerConfig(
            save_folder="/fsx/dev/jiaqi/checkpoints/olmo-3b-pretrain",
            max_duration=Duration.steps(60000),
            metrics_collect_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=2000),
        )
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("gc", GarbageCollectorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                project="a-evolve-olmo-3b",
                name="olmo-3b-pretrain",
            ),
        )
    )


# ═══════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════

def main():
    prepare_training_environment(seed=42)

    # 1. 构建模型（meta device，延迟初始化）
    model_config = build_model_config()
    model = model_config.build(init_device="meta")

    # 2. 构建训练模块（包含优化器 + FSDP 并行）
    train_module_config = build_train_module_config()
    train_module = train_module_config.build(model)

    # 3. 构建数据加载器
    data_loader = build_data_loader(train_module)

    # 4. 构建 Trainer 并启动训练
    trainer_config = build_trainer_config()
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    teardown_training_environment()


if __name__ == "__main__":
    main()
