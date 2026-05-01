"""Minimal end-to-end run of the ``a_evolve_training_multi`` algorithm.

Five user-owned role classes plug into ``run_cycle``. The training role
takes a ``TrainingJobRunner`` via constructor injection — swap the fake
backend below for ``h200_single_node`` / ``k8s_h200`` / ``sklearn_backend``
and nothing else changes.

Run::

    python examples/multi_role_min_example/drive_min_cycle.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_evolve.model.algorithms.a_evolve_training_multi import run_cycle
from agent_evolve.model.runner_protocol import TrainingJobRunner
from agent_evolve.model.types import (
    CheckpointRef,
    EvalMetrics,
    TrainingTrialResult,
)


# ── 1. A fake backend (stands in for any real ``TrainingJobRunner``) ──

class FakeBackend:
    """No-op ``TrainingJobRunner`` so the demo runs offline in milliseconds."""

    name = "fake_backend"

    def run_trial(self, workspace, node, budget, benchmark) -> TrainingTrialResult:
        ckpt_path = Path(workspace) / "fake_checkpoint.pt"
        ckpt_path.write_bytes(b"fake-weights")
        return TrainingTrialResult(
            node_id=getattr(node, "node_id", "demo"),
            workspace_path=str(workspace),
            status="success",
            checkpoint=CheckpointRef(name="demo", path=str(ckpt_path), kind="adapter"),
            eval_metrics=EvalMetrics(
                primary_metric_name="val_loss",
                primary_metric_value=0.42,
                maximize=False,
            ),
        )


# ── 2. Five user-owned role implementations ────────────────────────────
# Each is just a plain class with ``name`` + ``execute(my_dir, cycle_dir)``.
# No inheritance — the algorithm uses structural typing.

class DemoOrchestrator:
    name = "orchestrator"

    def execute(self, my_dir: Path, cycle_dir: Path) -> None:
        workspace = cycle_dir.parent.parent
        strategy = (workspace / "strategy.md").read_text()
        prior = sorted(
            p.name for p in cycle_dir.parent.iterdir()
            if p.is_dir() and p.name != cycle_dir.name and (p / "_done").exists()
        )
        (my_dir / "plan.md").write_text(
            f"# Plan for cycle {cycle_dir.name}\n\n"
            f"## Strategy in effect\n{strategy}\n"
            f"## Prior completed cycles\n{prior or '(none)'}\n\n"
            f"## Decision\nTrain one model with the default mix.\n"
        )


class DemoData:
    name = "data"

    def execute(self, my_dir: Path, cycle_dir: Path) -> None:
        # Could read ``cycle_dir / "orchestrator" / "plan.md"`` for guidance.
        (my_dir / "mix.yaml").write_text(
            "datasets:\n"
            "  - {name: tiny_corpus, weight: 1.0}\n"
        )


class DemoTraining:
    """TrainingRole with backend injected via constructor."""
    name = "training"

    def __init__(self, backend: TrainingJobRunner):
        self.backend = backend

    def execute(self, my_dir: Path, cycle_dir: Path) -> None:
        data_files = [p.name for p in (cycle_dir / "data").iterdir()]

        class _Node:
            node_id = cycle_dir.name

        result = self.backend.run_trial(
            workspace=my_dir,
            node=_Node(),
            budget=None,
            benchmark=None,
        )
        (my_dir / "checkpoint_pointer.json").write_text(
            json.dumps(
                {
                    "checkpoint": result.checkpoint.path if result.checkpoint else None,
                    "status": result.status,
                    "backend": self.backend.name,
                    "data_inputs_seen": data_files,
                },
                indent=2,
            )
        )


class DemoEvaluation:
    name = "evaluation"

    def execute(self, my_dir: Path, cycle_dir: Path) -> None:
        ptr = json.loads(
            (cycle_dir / "training" / "checkpoint_pointer.json").read_text()
        )
        (my_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "checkpoint_evaluated": ptr["checkpoint"],
                    "scores": {"val_loss": 0.42, "accuracy": 0.78},
                },
                indent=2,
            )
        )


class DemoAnalysis:
    name = "analysis"

    def execute(self, my_dir: Path, cycle_dir: Path) -> None:
        metrics = json.loads(
            (cycle_dir / "evaluation" / "metrics.json").read_text()
        )
        sibling_dirs = sorted(
            p.name for p in cycle_dir.iterdir() if p.is_dir()
        )
        cycles_so_far = sorted(
            p.name for p in cycle_dir.parent.iterdir()
            if p.is_dir() and (p / "_done").exists()
        )
        (my_dir / "summary.md").write_text(
            f"# {cycle_dir.name} summary\n\n"
            f"- val_loss: {metrics['scores']['val_loss']}\n"
            f"- accuracy: {metrics['scores']['accuracy']}\n"
            f"- siblings: {sibling_dirs}\n"
            f"- completed cycles before this one: {cycles_so_far}\n"
        )


# ── 3. Drive: 3 cycles, then dump the workspace tree ──────────────────

def main() -> None:
    workspace = Path("./_multi_role_min_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()
    (workspace / "strategy.md").write_text("# Strategy v1\nKeep it simple.\n")

    backend = FakeBackend()
    roles = [
        DemoOrchestrator(),
        DemoData(),
        DemoTraining(backend=backend),
        DemoEvaluation(),
        DemoAnalysis(),
    ]

    for _ in range(3):
        cycle_dir = run_cycle(workspace, roles)
        print(f"ran {cycle_dir.relative_to(workspace)}")

    print(f"\nworkspace tree under {workspace}:")
    for p in sorted(workspace.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(workspace)}")

    print("\nfinal analysis summary (cycles/0003/analysis/summary.md):")
    print((workspace / "cycles" / "0003" / "analysis" / "summary.md").read_text())


if __name__ == "__main__":
    main()
