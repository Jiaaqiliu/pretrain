"""Dynamic training configuration builder.

Generates OLMo-core compatible training configurations programmatically,
applying scaling laws, muTransfer, and user constraints.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autopilot.optimization.mu_transfer import ModelScale
from autopilot.utils.logging import get_logger

log = get_logger("experiment.config_builder")


class ModelSize(enum.Enum):
    """Predefined model sizes with canonical configurations."""

    TINY = "tiny"  # ~25M, for debugging
    SMALL = "small"  # ~190M, proxy models
    MEDIUM = "medium"  # ~1B, validation scale
    LARGE = "large"  # ~7B, production
    XL = "xl"  # ~13B
    XXL = "xxl"  # ~32B+

    def to_scale(self) -> ModelScale:
        configs = {
            ModelSize.TINY: ModelScale(hidden_size=256, num_layers=6, num_heads=4),
            ModelSize.SMALL: ModelScale(hidden_size=768, num_layers=12, num_heads=12),
            ModelSize.MEDIUM: ModelScale(hidden_size=2048, num_layers=24, num_heads=16),
            ModelSize.LARGE: ModelScale(hidden_size=4096, num_layers=32, num_heads=32),
            ModelSize.XL: ModelScale(hidden_size=5120, num_layers=40, num_heads=40),
            ModelSize.XXL: ModelScale(hidden_size=6656, num_layers=64, num_heads=52),
        }
        return configs[self]


class TrainingPhase(enum.Enum):
    PRETRAIN = "pretrain"
    MIDTRAIN = "midtrain"  # continued pretraining / domain adaptation
    SFT = "sft"  # supervised fine-tuning
    RLHF = "rlhf"  # reinforcement learning from human feedback


@dataclass
class ComputeBudget:
    """Compute resource constraints."""

    max_gpu_hours: Optional[float] = None
    num_nodes: int = 1
    gpus_per_node: int = 8
    gpu_type: str = "A100-80GB"
    max_wall_time_hours: Optional[float] = None

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @property
    def estimated_tflops_per_gpu(self) -> float:
        gpu_tflops = {
            "A100-40GB": 312,
            "A100-80GB": 312,
            "H100": 990,
            "H200": 990,
            "A10G": 125,
        }
        return gpu_tflops.get(self.gpu_type, 312)


@dataclass
class TrainingTarget:
    """High-level training objective specification."""

    model_size: ModelSize
    phase: TrainingPhase = TrainingPhase.PRETRAIN
    target_loss: Optional[float] = None
    target_tokens: Optional[int] = None
    compute_budget: Optional[ComputeBudget] = None
    data_domains: List[str] = field(default_factory=list)
    base_checkpoint: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedConfig:
    """A fully specified training configuration ready for submission."""

    name: str
    model_config: Dict[str, Any]
    optimizer_config: Dict[str, Any]
    scheduler_config: Dict[str, Any]
    data_config: Dict[str, Any]
    trainer_config: Dict[str, Any]
    launch_config: Dict[str, Any]
    overrides: List[str] = field(default_factory=list)

    def to_olmo_overrides(self) -> List[str]:
        """Convert to OLMo-core dotlist overrides."""
        overrides = []
        for key, value in self._flatten(self.optimizer_config, "train_module.optim"):
            overrides.append(f"{key}={value}")
        for key, value in self._flatten(self.scheduler_config, "train_module.scheduler"):
            overrides.append(f"{key}={value}")
        overrides.extend(self.overrides)
        return overrides

    def _flatten(self, d: Dict, prefix: str) -> List[tuple]:
        items = []
        for k, v in d.items():
            key = f"{prefix}.{k}"
            if isinstance(v, dict):
                items.extend(self._flatten(v, key))
            else:
                items.append((key, v))
        return items


class ConfigBuilder:
    """Builds training configurations from high-level targets.

    Applies:
    - Scaling laws for compute-optimal model/data sizing
    - Architecture best practices (from OLMo-2, Llama-3 reports)
    - Hardware-aware parallelism configuration
    - Learning rate and scheduler selection
    """

    def __init__(self):
        self._templates: Dict[str, Dict[str, Any]] = {}

    def build(self, target: TrainingTarget, params: Optional[Dict[str, Any]] = None) -> GeneratedConfig:
        """Build a complete training configuration from a target specification."""
        scale = target.model_size.to_scale()
        tokens = self._compute_optimal_tokens(scale, target)
        model_config = self._build_model_config(scale, target)
        optimizer_config = self._build_optimizer_config(scale, params)
        scheduler_config = self._build_scheduler_config(tokens, params)
        data_config = self._build_data_config(target)
        trainer_config = self._build_trainer_config(tokens, target)
        launch_config = self._build_launch_config(scale, target)

        name = self._generate_name(target, params)

        config = GeneratedConfig(
            name=name,
            model_config=model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            data_config=data_config,
            trainer_config=trainer_config,
            launch_config=launch_config,
        )

        log.info(
            f"Built config '{name}': {scale.num_params_approx/1e9:.2f}B params, "
            f"{tokens/1e9:.1f}B tokens, {launch_config.get('num_nodes', 1)} nodes"
        )
        return config

    def _compute_optimal_tokens(self, scale: ModelScale, target: TrainingTarget) -> int:
        """Compute optimal training tokens using Chinchilla scaling laws."""
        if target.target_tokens:
            return target.target_tokens

        # Chinchilla optimal: tokens ≈ 20 * params
        params = scale.num_params_approx
        chinchilla_tokens = int(20 * params)

        # If compute budget specified, adjust
        if target.compute_budget:
            budget = target.compute_budget
            total_tflops = (
                budget.total_gpus
                * budget.estimated_tflops_per_gpu
                * (budget.max_gpu_hours or 1000)
                * 3600
            )
            # FLOPs ≈ 6 * N * D
            max_tokens_from_budget = int(total_tflops * 1e12 / (6 * params))
            return min(chinchilla_tokens, max_tokens_from_budget)

        return chinchilla_tokens

    def _build_model_config(self, scale: ModelScale, target: TrainingTarget) -> Dict[str, Any]:
        """Build model architecture configuration."""
        return {
            "hidden_size": scale.hidden_size,
            "num_layers": scale.num_layers,
            "num_heads": scale.num_heads,
            "intermediate_size": scale.intermediate_size or int(scale.hidden_size * 8 / 3),
            "vocab_size": scale.vocab_size,
            "max_sequence_length": target.constraints.get("max_seq_len", 4096),
            "rope_theta": 500000,
            "attention_type": "flash",
            "activation": "swiglu",
            "norm_type": "rms_norm",
            "tie_embeddings": scale.num_params_approx < 3e9,
        }

    def _build_optimizer_config(
        self, scale: ModelScale, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build optimizer configuration with sensible defaults."""
        params = params or {}

        # Scale learning rate with model size (smaller models can use higher LR)
        default_lr = 3e-4 * (256 / scale.hidden_size) ** 0.5

        return {
            "type": params.get("optimizer", "adamw"),
            "lr": params.get("learning_rate", default_lr),
            "weight_decay": params.get("weight_decay", 0.1),
            "betas": [params.get("beta1", 0.9), params.get("beta2", 0.95)],
            "eps": 1e-8,
            "max_grad_norm": params.get("max_grad_norm", 1.0),
        }

    def _build_scheduler_config(
        self, total_tokens: int, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build learning rate scheduler configuration."""
        params = params or {}
        scheduler_type = params.get("scheduler", "cosine")

        warmup_steps = params.get("warmup_steps", 2000)

        if scheduler_type == "wsd":
            return {
                "type": "wsd",
                "warmup_steps": warmup_steps,
                "stable_fraction": 0.8,
                "decay_fraction": 0.2,
                "min_lr_ratio": 0.1,
            }
        elif scheduler_type == "cosine":
            return {
                "type": "cosine",
                "warmup_steps": warmup_steps,
                "min_lr_ratio": 0.1,
                "t_max": None,  # will be set to total steps
            }
        else:
            return {
                "type": "linear",
                "warmup_steps": warmup_steps,
                "min_lr_ratio": 0.0,
            }

    def _build_data_config(self, target: TrainingTarget) -> Dict[str, Any]:
        """Build data loading configuration."""
        return {
            "domains": target.data_domains or ["web", "code", "books", "academic"],
            "sequence_length": target.constraints.get("max_seq_len", 4096),
            "seed": 42,
        }

    def _build_trainer_config(self, total_tokens: int, target: TrainingTarget) -> Dict[str, Any]:
        """Build trainer configuration."""
        scale = target.model_size.to_scale()
        batch_tokens = self._compute_batch_size(scale, target)

        total_steps = total_tokens // batch_tokens

        return {
            "max_steps": total_steps,
            "total_tokens": total_tokens,
            "global_batch_tokens": batch_tokens,
            "checkpoint_interval": max(500, total_steps // 20),
            "eval_interval": max(500, total_steps // 40),
            "log_interval": 10,
            "save_folder": target.constraints.get("save_folder", None),
        }

    def _build_launch_config(self, scale: ModelScale, target: TrainingTarget) -> Dict[str, Any]:
        """Build job launch configuration based on model size and resources."""
        budget = target.compute_budget or ComputeBudget()

        # Auto-determine parallelism strategy
        params_b = scale.num_params_approx / 1e9
        if params_b < 1:
            dp_strategy = "ddp"
            tp_degree = 1
        elif params_b < 8:
            dp_strategy = "fsdp"
            tp_degree = 1
        elif params_b < 30:
            dp_strategy = "fsdp"
            tp_degree = 2
        else:
            dp_strategy = "hsdp"
            tp_degree = 4

        return {
            "num_nodes": budget.num_nodes,
            "gpus_per_node": budget.gpus_per_node,
            "dp_strategy": dp_strategy,
            "tp_degree": tp_degree,
            "gpu_type": budget.gpu_type,
        }

    def _compute_batch_size(self, scale: ModelScale, target: TrainingTarget) -> int:
        """Compute global batch size in tokens."""
        budget = target.compute_budget or ComputeBudget()
        seq_len = target.constraints.get("max_seq_len", 4096)

        # Heuristic: larger models benefit from larger batches
        # Start with ~2M tokens for 1B, scale up
        params_b = scale.num_params_approx / 1e9
        base_batch_tokens = int(2e6 * math.sqrt(max(1, params_b)))

        # Round to nice multiple of (gpus * seq_len)
        tokens_per_micro = seq_len * budget.total_gpus
        batch_tokens = max(tokens_per_micro, (base_batch_tokens // tokens_per_micro) * tokens_per_micro)

        return batch_tokens

    def _generate_name(self, target: TrainingTarget, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate a descriptive experiment name."""
        scale = target.model_size.to_scale()
        params_str = f"{scale.num_params_approx/1e6:.0f}M"
        phase = target.phase.value

        if params:
            lr = params.get("learning_rate", "")
            if lr:
                return f"autopilot-{params_str}-{phase}-lr{lr:.0e}"

        return f"autopilot-{params_str}-{phase}"
