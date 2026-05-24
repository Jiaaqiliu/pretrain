"""OLMo-3B 小规模训练验证：使用合成数据跑 200 步验证训练管线。"""

import sys
import os
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, "/fsx/dev/jiaqi/repos/OLMo-core/src")

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")

from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerDataParallelConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.train.callbacks import (
    ConsoleLoggerCallback,
    SpeedMonitorCallback,
    GarbageCollectorCallback,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train.common import Duration
from olmo_core.config import DType
from olmo_core.data.composable import (
    InMemoryTokenSource,
    ConcatAndChunkInstanceSource,
    ComposableDataLoaderConfig,
)

prepare_training_environment(seed=42)

# 1. 模型: OLMo-3B
model_config = TransformerConfig.olmo3_3B(vocab_size=100278)
model = model_config.build(init_device="meta")
print(f"Model built: olmo3_3B")

# 2. Train module: FSDP + AdamW
train_module_config = TransformerTrainModuleConfig(
    rank_microbatch_size=2 * 4096,
    max_sequence_length=4096,
    optim=AdamWConfig(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)),
    scheduler=CosWithWarmup(warmup_steps=20),
    max_grad_norm=1.0,
    dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp, param_dtype=DType.bfloat16),
    autocast_precision=DType.bfloat16,
)
train_module = train_module_config.build(model)
print("Train module built with FSDP")

# 3. 合成数据 (InMemoryTokenSource → ComposableDataLoader)
from olmo_core.data.tokenizer import TokenizerConfig

work_dir = Path("/fsx/dev/jiaqi/tmp/olmo-3b-smoke-data")
work_dir.mkdir(parents=True, exist_ok=True)

synthetic_tokens = np.random.randint(0, 100278, size=(500 * 4096,), dtype=np.uint32)
token_source = InMemoryTokenSource(synthetic_tokens, work_dir=work_dir)
instance_source = ConcatAndChunkInstanceSource(
    token_source, sequence_length=4096, work_dir=work_dir,
)

tok_cfg = TokenizerConfig(
    identifier="allenai/OLMo-2-0325-32B",
    vocab_size=100278,
    eos_token_id=100257,
    pad_token_id=100277,
)
loader_config = ComposableDataLoaderConfig(
    global_batch_size=16 * 4096,
    seed=42,
    tokenizer=tok_cfg,
    display_source_visualization=False,
)
data_loader = loader_config.build(
    instance_source,
    work_dir=work_dir,
    dp_process_group=train_module.dp_process_group,
)
print(f"Data loader built: global_batch_size={16*4096} tokens")

# 4. Trainer: 200 步
trainer_config = (
    TrainerConfig(
        save_folder="/fsx/dev/jiaqi/checkpoints/olmo-3b-smoke-v3",
        max_duration=Duration.steps(200),
        metrics_collect_interval=10,
    )
    .with_callback("console_logger", ConsoleLoggerCallback())
    .with_callback("speed_monitor", SpeedMonitorCallback())
    .with_callback("gc", GarbageCollectorCallback())
)

trainer = trainer_config.build(train_module, data_loader)
print("Trainer built. Starting fit()...")
trainer.fit()

teardown_training_environment()
print("SUCCESS: OLMo-3B trained 200 steps on 8 H200 GPUs!")
