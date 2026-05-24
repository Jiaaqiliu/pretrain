"""OLMo-3B Smoke Test: 50 steps on synthetic data to verify training works."""

import os
import sys
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")

try:
    import olmo_core
    print(f"olmo_core: {olmo_core.__version__}")
    HAS_OLMO = True
except ImportError:
    print("olmo_core not available, using pure PyTorch DDP smoke test")
    HAS_OLMO = False

if HAS_OLMO:
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.train import (
        TrainerConfig,
        prepare_training_environment,
        teardown_training_environment,
    )
    from olmo_core.train.train_module import TransformerTrainModuleConfig
    from olmo_core.train.train_module.transformer import TransformerDataParallelConfig
    from olmo_core.train.callbacks import ConsoleLoggerCallback, SpeedMonitorCallback
    from olmo_core.optim import AdamWConfig, CosWithWarmup
    from olmo_core.train.common import Duration
    from torch.utils.data import DataLoader, TensorDataset

    prepare_training_environment(seed=42)

    model_config = TransformerConfig(
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=100278,
        max_sequence_length=4096,
    )
    model = model_config.build(init_device="meta")

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=2 * 4096,
        max_sequence_length=4096,
        optim=AdamWConfig(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)),
        scheduler=CosWithWarmup(warmup_steps=10),
        max_grad_norm=1.0,
        dp_config=TransformerDataParallelConfig(),
    )
    train_module = train_module_config.build(model)

    dummy = torch.randint(0, 100278, (200, 4096))
    loader = DataLoader(TensorDataset(dummy), batch_size=2, shuffle=True)

    trainer_config = (
        TrainerConfig(
            save_folder="/fsx/dev/jiaqi/checkpoints/olmo-3b-smoke",
            max_duration=Duration.steps(50),
            metrics_collect_interval=5,
        )
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
    )

    trainer = trainer_config.build(train_module, loader)
    trainer.fit()
    teardown_training_environment()
    print("SUCCESS: OLMo-core 3B model trained 50 steps on 8 GPUs")

else:
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model = torch.nn.Linear(3072, 3072).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    for step in range(50):
        x = torch.randn(4, 3072, device="cuda")
        y = model(x)
        loss = y.mean()
        loss.backward()
        if step % 10 == 0 and rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    dist.destroy_process_group()
    if rank == 0:
        print("SUCCESS: PyTorch DDP smoke test completed 50 steps")
