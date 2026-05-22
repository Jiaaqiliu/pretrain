"""OLMo checkpoint loading utilities for thermodynamic measurement.

Supports loading OLMo checkpoints from:
  - Local filesystem (FSx)
  - HuggingFace Hub (allenai/OLMo-*)
  - OLMo-core native checkpoints

Provides batch discovery and parallel loading infrastructure.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class CheckpointInfo:
    """Metadata for a discovered checkpoint."""

    path: str
    step: int
    model_name: str
    model_size: str  # "190M", "1B", "7B", "13B"
    num_params: int
    lr_at_step: Optional[float] = None
    loss_at_step: Optional[float] = None
    batch_size: Optional[int] = None
    weight_decay: Optional[float] = None


# Known OLMo configurations
OLMO_CONFIGS = {
    "190M": {
        "hf_repo": "allenai/OLMo-2-0425-190M",
        "num_params": 190_000_000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "weight_decay": 0.1,
        "batch_size": 256,  # sequences
        "seq_len": 4096,
    },
    "1B": {
        "hf_repo": "allenai/OLMo-2-0425-1B",
        "num_params": 1_000_000_000,
        "hidden_size": 2048,
        "num_layers": 16,
        "num_heads": 16,
        "weight_decay": 0.1,
        "batch_size": 256,
        "seq_len": 4096,
    },
    "7B": {
        "hf_repo": "allenai/OLMo-2-0425-7B",
        "num_params": 7_000_000_000,
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "weight_decay": 0.1,
        "batch_size": 512,
        "seq_len": 4096,
    },
    "13B": {
        "hf_repo": "allenai/OLMo-2-0425-13B",
        "num_params": 13_000_000_000,
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "weight_decay": 0.1,
        "batch_size": 512,
        "seq_len": 4096,
    },
}


def discover_checkpoints(
    base_path: str,
    model_size: str,
    step_range: Optional[tuple[int, int]] = None,
    step_interval: Optional[int] = None,
) -> list[CheckpointInfo]:
    """Discover checkpoints on disk.

    Expects directory structure:
        base_path/step_NNNN/ or base_path/stepNNNN/

    Args:
        base_path: root checkpoint directory
        model_size: one of "190M", "1B", "7B", "13B"
        step_range: (min_step, max_step) filter
        step_interval: only return every N-th checkpoint
    """
    base = Path(base_path)
    if not base.exists():
        return []

    config = OLMO_CONFIGS.get(model_size, {})
    checkpoints = []

    step_pattern = re.compile(r"step[_-]?(\d+)")

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        match = step_pattern.match(entry.name)
        if not match:
            continue

        step = int(match.group(1))

        if step_range and (step < step_range[0] or step > step_range[1]):
            continue
        if step_interval and step % step_interval != 0:
            continue

        info = CheckpointInfo(
            path=str(entry),
            step=step,
            model_name=f"OLMo-{model_size}",
            model_size=model_size,
            num_params=config.get("num_params", 0),
            weight_decay=config.get("weight_decay", 0.1),
            batch_size=config.get("batch_size", 256),
        )

        # Try to load training metrics if available
        metrics_file = entry / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                info.loss_at_step = metrics.get("loss", metrics.get("train_loss"))
                info.lr_at_step = metrics.get("lr", metrics.get("learning_rate"))
            except (json.JSONDecodeError, OSError):
                pass

        checkpoints.append(info)

    return sorted(checkpoints, key=lambda c: c.step)


def discover_hf_checkpoints(
    model_size: str,
    cache_dir: Optional[str] = None,
) -> list[CheckpointInfo]:
    """Discover checkpoints from HuggingFace Hub.

    Uses the OLMo-2 revisions which provide intermediate checkpoints.
    """
    try:
        from huggingface_hub import list_repo_refs
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    config = OLMO_CONFIGS[model_size]
    repo_id = config["hf_repo"]

    refs = list_repo_refs(repo_id)
    checkpoints = []

    step_pattern = re.compile(r"step(\d+)(?:-tokens.*)?")

    for branch in refs.branches:
        match = step_pattern.match(branch.name)
        if not match:
            continue

        step = int(match.group(1))
        info = CheckpointInfo(
            path=f"hf://{repo_id}@{branch.name}",
            step=step,
            model_name=f"OLMo-{model_size}",
            model_size=model_size,
            num_params=config["num_params"],
            weight_decay=config["weight_decay"],
            batch_size=config["batch_size"],
        )
        checkpoints.append(info)

    return sorted(checkpoints, key=lambda c: c.step)


def load_olmo_checkpoint(
    checkpoint_info: CheckpointInfo,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Load an OLMo model from checkpoint.

    Supports:
      - Local OLMo-core format (step_NNNN/)
      - HuggingFace format (hf://repo@revision)
      - SafeTensors format
    """
    path = checkpoint_info.path

    if path.startswith("hf://"):
        return _load_from_hf(path, device, dtype)
    else:
        return _load_from_local(path, checkpoint_info.model_size, device, dtype)


def _load_from_hf(
    hf_path: str, device: str, dtype: torch.dtype
) -> torch.nn.Module:
    """Load model from HuggingFace Hub."""
    from transformers import AutoModelForCausalLM

    # Parse hf://repo@revision
    parts = hf_path.replace("hf://", "").split("@")
    repo_id = parts[0]
    revision = parts[1] if len(parts) > 1 else None

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    return model


def _load_from_local(
    path: str, model_size: str, device: str, dtype: torch.dtype
) -> torch.nn.Module:
    """Load model from local OLMo-core checkpoint."""
    checkpoint_dir = Path(path)

    # Try OLMo-core native format first
    model_pt = checkpoint_dir / "model.pt"
    if model_pt.exists():
        return _load_olmo_core_native(checkpoint_dir, model_size, device, dtype)

    # Try safetensors format
    safetensors_files = list(checkpoint_dir.glob("*.safetensors"))
    if safetensors_files:
        return _load_safetensors(checkpoint_dir, device, dtype)

    # Try HF format from local directory
    config_json = checkpoint_dir / "config.json"
    if config_json.exists():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        return model

    raise FileNotFoundError(
        f"Could not find loadable checkpoint at {path}. "
        f"Expected model.pt, *.safetensors, or config.json"
    )


def _load_olmo_core_native(
    checkpoint_dir: Path, model_size: str, device: str, dtype: torch.dtype
) -> torch.nn.Module:
    """Load from OLMo-core native format using the OLMo-core library."""
    import sys

    olmo_core_src = Path(__file__).resolve().parents[2] / "olmo-core" / "src"
    if olmo_core_src.exists() and str(olmo_core_src) not in sys.path:
        sys.path.insert(0, str(olmo_core_src))

    from olmo_core.nn.transformer import TransformerConfig

    factory_map = {
        "190M": "olmo2_190M",
        "1B": "olmo2_1B",
        "7B": "olmo2_7B",
        "13B": "olmo2_13B",
    }

    factory_name = factory_map.get(model_size)
    if factory_name is None:
        raise ValueError(f"Unknown model size: {model_size}")

    config = getattr(TransformerConfig, factory_name)()
    model = config.build(init_device="cpu")

    state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    return model


def _load_safetensors(
    checkpoint_dir: Path, device: str, dtype: torch.dtype
) -> torch.nn.Module:
    """Load from safetensors format."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    return model


def load_training_logs(
    log_path: str,
) -> dict[int, dict]:
    """Load training logs to extract loss, LR, gradient stats per step.

    Supports multiple formats:
      - WandB export (JSON)
      - OLMo-core metrics (metrics.json per checkpoint)
      - CSV format (step, loss, lr, ...)

    Returns:
        dict mapping step → {loss, lr, grad_norm, grad_variance, ...}
    """
    path = Path(log_path)
    logs = {}

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                step = entry.get("step", entry.get("_step"))
                if step is not None:
                    logs[int(step)] = entry
        elif isinstance(data, dict):
            logs = {int(k): v for k, v in data.items()}

    elif path.suffix == ".csv":
        import csv

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row.get("step", row.get("global_step", 0)))
                logs[step] = {k: _try_float(v) for k, v in row.items()}

    elif path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                step = entry.get("step", entry.get("global_step"))
                if step is not None:
                    logs[int(step)] = entry

    return logs


def _try_float(val):
    """Try to convert to float, return original if fails."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return val
