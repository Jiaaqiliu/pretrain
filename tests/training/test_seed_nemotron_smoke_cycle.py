"""PR8: end-to-end smoke — TrainingEvolver(...).run(cycles=1) against the seed."""

from __future__ import annotations

import json
from pathlib import Path

import agent_evolve as ae
from agent_evolve.training.types import TrainingEvolveConfig


SEED = Path(__file__).resolve().parents[2] / "seed_workspaces" / "nemotron_reasoner"


def test_training_evolver_run_cycles_1(tmp_path: Path) -> None:
    evolver = ae.TrainingEvolver(
        workspace=SEED,
        benchmark="nemo_reasoner",
        algorithm="mcgs",
        backend="h200_single_node",
        config=TrainingEvolveConfig(smoke=True, max_cycles=1, trial_budget_seconds=30.0),
        work_dir=tmp_path / "smoke",
    )
    result = evolver.run(cycles=1)
    assert result.cycles_completed == 1
    # Graph persisted.
    assert Path(result.graph_path).exists()
    with open(result.graph_path) as f:
        graph = json.load(f)
    # Root + one candidate.
    assert len(graph["nodes"]) >= 2
    # Report file for cycle 1 exists.
    reports_dir = Path(evolver.workspace.root) / "evolution" / "reports"
    assert (reports_dir / "cycle_0001.json").exists()


def test_failed_trial_is_preserved(tmp_path: Path) -> None:
    """If the single trial is invalid, the node still lives in the graph."""
    evolver = ae.TrainingEvolver(
        workspace=SEED,
        benchmark="nemo_reasoner",
        algorithm="mcgs",
        backend="h200_single_node",
        config=TrainingEvolveConfig(smoke=True, max_cycles=1, trial_budget_seconds=30.0),
        work_dir=tmp_path / "smoke",
    )
    result = evolver.run(cycles=1)
    with open(result.graph_path) as f:
        graph = json.load(f)
    # Every non-root node has a reward (positive or negative), never None.
    for node in graph["nodes"]:
        if node["node_id"] == "node-root":
            continue
        assert node["reward"] is not None
