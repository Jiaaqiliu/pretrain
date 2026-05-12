"""Training workspace structural validation.

A training workspace is a tree whose evolvable surface is declared by
``manifest.yaml``. Structural validation is therefore manifest-driven:
required files are the ``evolvable_layers`` + ``protected_layers`` the
manifest itself declares. Artifact layers are created-if-missing since
they hold per-cycle runtime state.

Supported contract versions:
  train-1.0  — legacy MCGS layout (model/, data/mix.yaml, train/pipeline.yaml).
  train-1.1  — fork-as-world layout (recipes/{train,data}/).
"""

from __future__ import annotations

from pathlib import Path


SUPPORTED_CONTRACT_VERSIONS = {"train-1.0", "train-1.1"}

# Legacy (v1.0) hard-coded layout checks. Kept so existing v1.0 seed
# workspaces (arc-mas, swe, terminal, etc.) continue to validate.
_V1_0_REQUIRED_FILES = [
    "manifest.yaml",
    "model/base.yaml",
    "model/adapter.yaml",
    "data/sources.yaml",
    "data/mix.yaml",
    "train/pipeline.yaml",
    "eval/local_splits.yaml",
    "eval/error_taxonomy.yaml",
]
_V1_0_REQUIRED_DIRS = [
    "model", "data", "train", "eval",
    "memory", "checkpoints", "evolution",
]


def validate_training_workspace(root: str | Path) -> list[str]:
    """Return a list of structural errors. Empty list means valid."""
    root = Path(root)
    errors: list[str] = []

    if not root.is_dir():
        return [f"Workspace root does not exist: {root}"]

    manifest = root / "manifest.yaml"
    if not manifest.exists():
        return ["Missing manifest.yaml"]

    try:
        import yaml
        with open(manifest) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        return [f"Failed to parse manifest.yaml: {exc}"]

    if "name" not in raw:
        errors.append("manifest.yaml missing required field: name")

    cv = raw.get("contract_version") or "train-1.0"
    if cv not in SUPPORTED_CONTRACT_VERSIONS:
        errors.append(
            f"Contract version {cv!r} not supported "
            f"(expected one of {sorted(SUPPORTED_CONTRACT_VERSIONS)})"
        )
        return errors

    if cv == "train-1.0":
        for rel in _V1_0_REQUIRED_FILES:
            if not (root / rel).exists():
                errors.append(f"Missing required file: {rel}")
        for d in _V1_0_REQUIRED_DIRS:
            target = root / d
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
        return errors

    # train-1.1: manifest-driven. Evolvable + protected layers (if they
    # name specific files) must exist; artifact layers are created empty.
    for rel in raw.get("evolvable_layers", []) + raw.get("protected_layers", []):
        target = root / rel
        if not target.exists():
            errors.append(f"Missing layer: {rel}")
    for rel in raw.get("artifact_layers", []):
        target = root / rel
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
    return errors
