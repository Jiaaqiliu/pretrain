"""PR8: the shipped nemotron_reasoner seed workspace validates cleanly."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.training.schema import validate_training_workspace


SEED = Path(__file__).resolve().parents[2] / "seed_workspaces" / "nemotron_reasoner"


def test_schema_validates() -> None:
    errors = validate_training_workspace(SEED)
    assert errors == [], errors


def test_manifest_contract_version() -> None:
    import yaml

    with open(SEED / "manifest.yaml") as f:
        manifest = yaml.safe_load(f)
    assert manifest["contract_version"] == "train-1.0"
    assert manifest["defaults"]["benchmark"] == "nemo_reasoner"
    assert manifest["defaults"]["algorithm"] == "mcgs"
    assert manifest["defaults"]["backend"] == "h200_single_node"
