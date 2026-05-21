"""Translates A-EVOLVE workspace YAML configs into OLMo-core config objects.

This is the bridge between the workspace-on-disk format (train/optimizer.yaml,
model/base.yaml, data/sources.yaml, etc.) and the olmo_core Config system
(TransformerConfig, AdamWConfig, TrainerConfig, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class OLMoCoreTrainingConfig:
    """Intermediate representation of an OLMo-core training configuration.

    Generated from A-EVOLVE workspace files, then used by ScriptGenerator
    to produce a runnable training script.
    """

    # Model architecture
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50304
    max_sequence_length: int = 4096
    rope_theta: float = 500000.0
    model_path: str | None = None

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1

    # Training
    max_steps: int = 10000
    global_batch_size: int = 256
    rank_microbatch_size: int = 4096
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10
    save_folder: str = "./checkpoints"

    # Data
    data_paths: list[str] = field(default_factory=list)
    data_mix_ratios: dict[str, float] = field(default_factory=dict)
    tokenizer_path: str | None = None

    # Distributed
    dp_degree: int | None = None
    tp_degree: int = 1
    pp_degree: int = 1
    fsdp: bool = True

    # Launch
    num_nodes: int = 1
    gpus_per_node: int = 8

    # Callbacks
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    metric_report_url: str | None = None

    # Phase
    phase: str = "pretrain"
    load_path: str | None = None


class OLMoCoreConfigTranslator:
    """Translates A-EVOLVE workspace layout to OLMoCoreTrainingConfig."""

    def translate(self, workspace_root: Path) -> OLMoCoreTrainingConfig:
        """Read workspace YAML files and produce an OLMoCoreTrainingConfig."""
        config = OLMoCoreTrainingConfig()

        model_cfg = self._load_yaml(workspace_root / "model" / "base.yaml")
        optimizer_cfg = self._load_yaml(workspace_root / "train" / "optimizer.yaml")
        batching_cfg = self._load_yaml(workspace_root / "train" / "batching.yaml")
        pipeline_cfg = self._load_yaml(workspace_root / "train" / "pipeline.yaml")
        data_cfg = self._load_yaml(workspace_root / "data" / "sources.yaml")
        mix_cfg = self._load_yaml(workspace_root / "data" / "mix.yaml")

        self._apply_model(config, model_cfg)
        self._apply_optimizer(config, optimizer_cfg)
        self._apply_batching(config, batching_cfg)
        self._apply_pipeline(config, pipeline_cfg)
        self._apply_data(config, data_cfg, mix_cfg, workspace_root)

        return config

    def _apply_model(self, config: OLMoCoreTrainingConfig, model_cfg: dict) -> None:
        if not model_cfg:
            return
        config.model_path = model_cfg.get("path")
        arch = model_cfg.get("architecture", {})
        if arch:
            config.d_model = arch.get("hidden_size", config.d_model)
            config.n_layers = arch.get("num_layers", config.n_layers)
            config.n_heads = arch.get("num_heads", config.n_heads)
            config.vocab_size = arch.get("vocab_size", config.vocab_size)
            config.max_sequence_length = arch.get("max_seq_len", config.max_sequence_length)
        # Flat format (some workspaces use this)
        if "hidden_size" in model_cfg:
            config.d_model = model_cfg["hidden_size"]
        if "num_layers" in model_cfg:
            config.n_layers = model_cfg["num_layers"]
        if "num_heads" in model_cfg:
            config.n_heads = model_cfg["num_heads"]

    def _apply_optimizer(self, config: OLMoCoreTrainingConfig, optim_cfg: dict) -> None:
        if not optim_cfg:
            return
        config.lr = optim_cfg.get("lr", config.lr)
        config.weight_decay = optim_cfg.get("weight_decay", config.weight_decay)
        betas = optim_cfg.get("betas")
        if betas and len(betas) == 2:
            config.betas = tuple(betas)
        config.eps = optim_cfg.get("eps", config.eps)
        config.max_grad_norm = optim_cfg.get("max_grad_norm", config.max_grad_norm)
        config.warmup_steps = optim_cfg.get("warmup_steps", config.warmup_steps)
        config.warmup_steps = optim_cfg.get("warmup_ratio_steps", config.warmup_steps)

    def _apply_batching(self, config: OLMoCoreTrainingConfig, batch_cfg: dict) -> None:
        if not batch_cfg:
            return
        per_device = batch_cfg.get("per_device_train_batch_size", 4)
        grad_accum = batch_cfg.get("gradient_accumulation_steps", 1)
        seq_len = batch_cfg.get("max_seq_len", config.max_sequence_length)
        config.rank_microbatch_size = per_device * seq_len
        config.max_sequence_length = seq_len
        config.log_interval = batch_cfg.get("log_every", config.log_interval)

    def _apply_pipeline(self, config: OLMoCoreTrainingConfig, pipeline_cfg: dict) -> None:
        if not pipeline_cfg:
            return
        stages = pipeline_cfg.get("stages", [])
        for stage in stages:
            if not stage.get("enabled", True):
                continue
            stype = stage.get("type", "")
            if stype == "sft":
                config.phase = "sft"
                config.max_steps = stage.get("max_steps", config.max_steps)
                epochs = stage.get("epochs")
                if epochs and not stage.get("max_steps"):
                    config.max_steps = int(epochs * 10000)
            elif stype == "pretrain":
                config.phase = "pretrain"
                config.max_steps = stage.get("max_steps", config.max_steps)

    def _apply_data(
        self,
        config: OLMoCoreTrainingConfig,
        data_cfg: dict | list,
        mix_cfg: dict,
        workspace_root: Path,
    ) -> None:
        if isinstance(data_cfg, list):
            for src in data_cfg:
                path = src.get("path", "")
                if path:
                    config.data_paths.append(str(path))
        elif isinstance(data_cfg, dict):
            sources = data_cfg.get("sources", [])
            for src in sources:
                path = src.get("path", "")
                if path:
                    config.data_paths.append(str(path))

        if mix_cfg:
            ratios = mix_cfg.get("ratios", mix_cfg.get("weights", {}))
            if isinstance(ratios, dict):
                config.data_mix_ratios = ratios

    def _load_yaml(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to parse %s: %s", path, e)
            return {}
