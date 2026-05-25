"""Bridge between autopretrain search nodes and OLMo-core workspace format.

Converts a CurriculumSchedule + SearchConfig into the workspace YAML structure
that the OLMoCoreBackend expects, then calls the backend to run a trial.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import yaml

from .mixture import CurriculumSchedule, DataMixture

logger = logging.getLogger(__name__)

WORKSPACE_TEMPLATE = "olmo_core_pretrain"


class WorkspaceBridge:
    """Creates ephemeral workspace directories for each search trial."""

    # Map from logical domain names to actual data paths on FSx
    DOMAIN_PATHS = {
        "web": "/fsx/dev/jiaqi/data/olmo-pretrain/dclm_web",
        "code": "/fsx/dev/jiaqi/data/olmo-pretrain/code",
        "math": "/fsx/dev/jiaqi/data/olmo-pretrain/math",
        "academic": "/fsx/dev/jiaqi/data/olmo-pretrain/fineweb_edu",
    }

    def __init__(
        self,
        *,
        template_workspace: Path | None = None,
        trials_root: Path = Path("./autopretrain_trials"),
        data_root: str = "/fsx/dev/jiaqi/data/olmo-pretrain",
        domain_paths: dict[str, str] | None = None,
    ):
        if template_workspace is None:
            template_workspace = (
                Path(__file__).resolve().parents[4]
                / "seed_workspaces"
                / "olmo_core_pretrain"
            )
        self.template = template_workspace
        self.trials_root = trials_root
        self.data_root = data_root
        self.domain_paths = domain_paths or self.DOMAIN_PATHS

    def create_trial_workspace(
        self,
        node_id: str,
        curriculum: CurriculumSchedule,
        *,
        model_factory: str = "olmo2_190M",
        max_steps: int = 5000,
        batch_size: int = 64,
        sequence_length: int = 4096,
        lr: float = 3e-4,
        warmup_steps: int = 500,
    ) -> Path:
        """Create a workspace directory for a single trial.

        Copies the template workspace and overwrites data/mix.yaml and
        train/pipeline.yaml with values from the curriculum.
        """
        trial_dir = self.trials_root / node_id
        if trial_dir.exists():
            shutil.rmtree(trial_dir)

        shutil.copytree(self.template, trial_dir)

        # Write model config
        self._write_model_config(trial_dir, model_factory, sequence_length)

        # Write data sources
        self._write_data_sources(trial_dir, curriculum)

        # Write data mix (from first phase or static)
        self._write_data_mix(trial_dir, curriculum)

        # Write training pipeline
        self._write_pipeline(trial_dir, max_steps)

        # Write optimizer
        self._write_optimizer(trial_dir, lr, warmup_steps)

        # Write batching
        self._write_batching(trial_dir, batch_size, sequence_length)

        # Write curriculum schedule (custom file for multi-phase)
        if not curriculum.is_static:
            self._write_curriculum(trial_dir, curriculum)

        logger.debug("Created trial workspace at %s", trial_dir)
        return trial_dir

    def _write_model_config(self, trial_dir: Path, factory: str, seq_len: int):
        model_config = {
            "name": factory.replace("_", "-").upper(),
            "path": None,
            "factory": factory,
            "architecture": {
                "max_seq_len": seq_len,
            },
            "dtype": "bfloat16",
        }
        (trial_dir / "model" / "base.yaml").write_text(
            yaml.dump(model_config, default_flow_style=False)
        )

    def _write_data_sources(self, trial_dir: Path, curriculum: CurriculumSchedule):
        mixture = curriculum.phases[0].mixture
        sources = []
        for domain in mixture.domains:
            path = self.domain_paths.get(domain, f"{self.data_root}/{domain}")
            sources.append({
                "path": path,
                "split": "train",
                "format": "numpy_uint32",
                "domain": domain,
            })
        (trial_dir / "data" / "sources.yaml").write_text(
            yaml.dump({"sources": sources}, default_flow_style=False)
        )

    def _write_data_mix(self, trial_dir: Path, curriculum: CurriculumSchedule):
        # For static or first phase of curriculum
        mixture = curriculum.phases[0].mixture
        mix_config = {"ratios": dict(mixture.weights)}

        # Include filter config
        if mixture.filter_config.retention_rates:
            mix_config["filter_retention"] = dict(mixture.filter_config.retention_rates)

        (trial_dir / "data" / "mix.yaml").write_text(
            yaml.dump(mix_config, default_flow_style=False)
        )

    def _write_pipeline(self, trial_dir: Path, max_steps: int):
        pipeline = {
            "stages": [{
                "name": "proxy_pretrain",
                "type": "pretrain",
                "enabled": True,
                "max_steps": max_steps,
                "checkpoint_interval": max_steps,  # Only save at end for proxy
                "eval_interval": max_steps,
            }]
        }
        (trial_dir / "train" / "pipeline.yaml").write_text(
            yaml.dump(pipeline, default_flow_style=False)
        )

    def _write_optimizer(self, trial_dir: Path, lr: float, warmup_steps: int):
        optimizer = {
            "lr": lr,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "max_grad_norm": 1.0,
            "warmup_steps": warmup_steps,
        }
        (trial_dir / "train" / "optimizer.yaml").write_text(
            yaml.dump(optimizer, default_flow_style=False)
        )

    def _write_batching(self, trial_dir: Path, batch_size: int, seq_len: int):
        # batch_size here is in sequences
        # For proxy: smaller batch to fit on fewer GPUs
        batching = {
            "per_device_train_batch_size": max(1, batch_size // 8),
            "gradient_accumulation_steps": 1,
            "max_seq_len": seq_len,
            "log_every": 50,
        }
        (trial_dir / "train" / "batching.yaml").write_text(
            yaml.dump(batching, default_flow_style=False)
        )

    def _write_curriculum(self, trial_dir: Path, curriculum: CurriculumSchedule):
        """Write multi-phase curriculum as a separate config file."""
        curriculum_config = curriculum.to_dict()
        (trial_dir / "data" / "curriculum.yaml").write_text(
            yaml.dump(curriculum_config, default_flow_style=False)
        )
