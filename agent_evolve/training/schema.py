"""Training workspace structural validation."""

from __future__ import annotations

from pathlib import Path


TRAINING_CONTRACT_VERSION = "train-1.0"

REQUIRED_FILES: list[str] = [
    "manifest.yaml",
    "model/base.yaml",
    "model/adapter.yaml",
    "data/sources.yaml",
    "data/mix.yaml",
    "train/pipeline.yaml",
    "eval/local_splits.yaml",
    "eval/error_taxonomy.yaml",
]

REQUIRED_DIRS: list[str] = [
    "model",
    "data",
    "train",
    "eval",
    "memory",
    "checkpoints",
    "evolution",
]


def validate_training_workspace(root: str | Path) -> list[str]:
    """Return a list of structural errors. Empty list means valid."""
    root = Path(root)
    errors: list[str] = []

    if not root.is_dir():
        return [f"Workspace root does not exist: {root}"]

    manifest = root / "manifest.yaml"
    if not manifest.exists():
        errors.append("Missing manifest.yaml")
    else:
        try:
            import yaml

            with open(manifest) as f:
                raw = yaml.safe_load(f) or {}
            if "name" not in raw:
                errors.append("manifest.yaml missing required field: name")
            cv = raw.get("contract_version")
            if cv and cv != TRAINING_CONTRACT_VERSION:
                errors.append(
                    f"Contract version mismatch: got {cv}, expected {TRAINING_CONTRACT_VERSION}"
                )
        except Exception as exc:
            errors.append(f"Failed to parse manifest.yaml: {exc}")

    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            errors.append(f"Missing required file: {rel}")

    # Create optional directories that are expected to exist (artifact layers).
    for d in REQUIRED_DIRS:
        target = root / d
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)

    return errors
