"""OLMo-core integration bridge.

Provides direct integration with OLMo-core's training infrastructure:
- Generate real OLMo-core Config objects from AutoPilot's GeneratedConfig
- Inject monitoring callbacks into OLMo-core Trainer
- Use OLMo-core's BeakerLaunchConfig for job submission
- Access OLMo-core's data loading and checkpoint systems
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from autopilot.experiment.config_builder import GeneratedConfig
from autopilot.utils.logging import get_logger

log = get_logger("integrations.olmo_core")


class OLMoCoreConfigGenerator:
    """Generates OLMo-core compatible Python training scripts and configs.

    Instead of submitting arbitrary commands, this generates actual OLMo-core
    training scripts that use the Config/Trainer/TrainModule API.
    """

    def generate_training_script(self, config: GeneratedConfig) -> str:
        """Generate a complete OLMo-core training script from configuration."""
        model = config.model_config
        optim = config.optimizer_config
        scheduler = config.scheduler_config
        trainer = config.trainer_config

        script = f'''"""AutoPilot-generated training script: {config.name}"""

from olmo_core.config import Config
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import TrainerConfig
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConsoleLoggerCallback,
    WandBCallback,
    SpeedMonitorCallback,
    GarbageCollectorCallback,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.internal.experiment import SubCmd, build_config


def build_model_config() -> TransformerConfig:
    return TransformerConfig(
        d_model={model['hidden_size']},
        n_layers={model['num_layers']},
        n_heads={model['num_heads']},
        vocab_size={model.get('vocab_size', 50304)},
        max_sequence_length={model.get('max_sequence_length', 4096)},
    )


def build_train_module_config(model_config: TransformerConfig) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        model=model_config,
        optim=AdamWConfig(
            lr={optim['lr']},
            weight_decay={optim['weight_decay']},
            betas=tuple({optim.get('betas', [0.9, 0.95])}),
        ),
        scheduler=CosWithWarmup(
            warmup_steps={scheduler.get('warmup_steps', 2000)},
        ),
        max_grad_norm={optim.get('max_grad_norm', 1.0)},
        rank_microbatch_size={model.get('max_sequence_length', 4096)},
    )


def build_trainer_config(
    train_module_config: TransformerTrainModuleConfig,
) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder="{trainer.get('save_folder', '/checkpoints/' + config.name)}",
            max_duration={trainer['max_steps']},
            metrics_collect_interval={trainer.get('log_interval', 10)},
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval={trainer.get('checkpoint_interval', 1000)}),
        )
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("gc", GarbageCollectorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                project="autopilot",
                name="{config.name}",
            ),
        )
    )


if __name__ == "__main__":
    model_config = build_model_config()
    train_module_config = build_train_module_config(model_config)
    trainer_config = build_trainer_config(train_module_config)

    trainer = trainer_config.build(train_module_config)
    trainer.fit()
'''
        return script

    def generate_launch_config(self, config: GeneratedConfig) -> str:
        """Generate OLMo-core BeakerLaunchConfig code."""
        launch = config.launch_config
        return f'''
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage

launch_config = BeakerLaunchConfig(
    name="{config.name}",
    num_nodes={launch.get('num_nodes', 1)},
    num_gpus={launch.get('gpus_per_node', 8)},
    beaker_image=OLMoCoreBeakerImage.stable,
    torchrun=True,
    cmd=[
        "src/scripts/autopilot/{config.name}.py",
        "train",
        "{config.name}",
    ],
)

launch_config.launch()
'''

    def generate_callback_injection(self) -> str:
        """Generate code for AutoPilot's monitoring callback injected into OLMo-core."""
        return '''
"""AutoPilot monitoring callback for OLMo-core Trainer."""

import time
import json
import httpx
from olmo_core.train.callbacks import Callback


class AutoPilotMonitorCallback(Callback):
    """Reports metrics to the AutoPilot agent for monitoring and decision-making."""

    def __init__(self, agent_url: str = "http://localhost:8765", experiment_id: str = ""):
        self._agent_url = agent_url
        self._experiment_id = experiment_id
        self._step = 0

    def post_step(self):
        self._step = self.trainer.global_step

    def log_metrics(self, metrics: dict):
        payload = {
            "experiment_id": self._experiment_id,
            "step": self._step,
            "timestamp": time.time(),
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        try:
            httpx.post(f"{self._agent_url}/metrics", json=payload, timeout=2.0)
        except Exception:
            pass  # non-blocking: agent might be temporarily unavailable

    def on_error(self, exc: Exception):
        payload = {
            "experiment_id": self._experiment_id,
            "step": self._step,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        try:
            httpx.post(f"{self._agent_url}/error", json=payload, timeout=5.0)
        except Exception:
            pass
'''


class OLMoCoreBridge:
    """High-level bridge between AutoPilot and OLMo-core.

    Handles:
    - Converting AutoPilot configs to OLMo-core training scripts
    - Submitting jobs via OLMo-core's native launch system
    - Collecting metrics via injected callbacks
    """

    def __init__(self, repo_path: Optional[str] = None):
        self._repo_path = repo_path
        self._generator = OLMoCoreConfigGenerator()
        self._available = self._check_olmo_core_available()

    @property
    def available(self) -> bool:
        return self._available

    def generate_experiment(self, config: GeneratedConfig) -> Dict[str, str]:
        """Generate all files needed for an OLMo-core training experiment."""
        files = {
            f"src/scripts/autopilot/{config.name}.py": self._generator.generate_training_script(config),
            f"src/scripts/autopilot/launch_{config.name}.py": self._generator.generate_launch_config(config),
        }
        return files

    def get_olmo_core_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return predefined OLMo-core model configurations."""
        return {
            "OLMo2-190M": {
                "factory": "olmo2_190M",
                "params": 190e6,
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
            },
            "OLMo2-370M": {
                "factory": "olmo2_370M",
                "params": 370e6,
                "hidden_size": 1024,
                "num_layers": 16,
                "num_heads": 16,
            },
            "OLMo2-1B": {
                "factory": "olmo2_1B",
                "params": 1e9,
                "hidden_size": 2048,
                "num_layers": 16,
                "num_heads": 16,
            },
            "OLMo2-7B": {
                "factory": "olmo2_7B",
                "params": 7e9,
                "hidden_size": 4096,
                "num_layers": 32,
                "num_heads": 32,
            },
            "OLMo2-13B": {
                "factory": "olmo2_13B",
                "params": 13e9,
                "hidden_size": 5120,
                "num_layers": 40,
                "num_heads": 40,
            },
            "OLMo2-32B": {
                "factory": "olmo2_32B",
                "params": 32e9,
                "hidden_size": 5120,
                "num_layers": 64,
                "num_heads": 40,
            },
        }

    def _check_olmo_core_available(self) -> bool:
        try:
            import olmo_core  # noqa: F401
            return True
        except ImportError:
            return False
