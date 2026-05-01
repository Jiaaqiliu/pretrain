"""The 4+1 fixed roles.

Names and default execution order are the only role-level facts the
platform guarantees. Each implementation here is a stub: it writes a
single ``TODO.md`` so a fresh workspace doesn't crash. Each owning team
replaces the stub with whatever they want — same ``name``, same
``execute`` signature, anything else free.

The five role names are:

* ``orchestrator`` — the TechLead. Plans the cycle.
* ``data``         — selects / mixes / filters datasets.
* ``training``     — runs the training step.
* ``evaluation``   — evaluates the resulting checkpoint.
* ``analysis``     — reads everything and writes report + dashboard.
"""
from __future__ import annotations

from pathlib import Path


class _StubRole:
    """Placeholder writing a TODO marker. Each team owns its replacement."""

    name: str = ""
    description: str = ""

    def execute(self, my_dir: Path, cycle_dir: Path) -> None:
        siblings = sorted(p.name for p in cycle_dir.iterdir() if p.is_dir())
        content = (
            f"# {self.name} — not implemented\n\n"
            f"{self.description}\n\n"
            "Owner team: replace this stub with a real implementation.\n"
            "Anything written into this directory is your responsibility:\n"
            "file names, formats, and schemas are unconstrained.\n\n"
            f"Cycle root: {cycle_dir}\n"
            f"Sibling roles: {siblings}\n"
        )
        (my_dir / "TODO.md").write_text(content)


class OrchestratorRole(_StubRole):
    """The TechLead. Reads ``strategy.md`` + prior cycles, writes the plan."""

    name = "orchestrator"
    description = (
        "Decide what this cycle should accomplish. Read `strategy.md` from\n"
        "the workspace root and any prior `cycles/*/` to inform your plan.\n"
        "Write the strategy update + per-role intent here. Format is your\n"
        "choice (md, jsonl, yaml, ...)."
    )


class DataRole(_StubRole):
    """Owns dataset selection / mixing / filtering for this cycle."""

    name = "data"
    description = (
        "Decide what data goes into this cycle's training run. Read\n"
        "`../orchestrator/` for what was asked. Write the manifest /\n"
        "mix / filters / processed-pointer here. Format is your choice."
    )


class TrainingRole(_StubRole):
    """Runs the training step. Reads the data manifest from upstream."""

    name = "training"
    description = (
        "Run training. Read `../data/` and `../orchestrator/` for inputs.\n"
        "Write checkpoint pointers, logs, and a summary here. Format is\n"
        "your choice."
    )


class EvaluationRole(_StubRole):
    """Runs evaluation against the upstream checkpoint."""

    name = "evaluation"
    description = (
        "Evaluate the checkpoint produced by `../training/`. Write\n"
        "metrics and any benchmark output here. Format is your choice."
    )


class AnalysisRole(_StubRole):
    """Reads the whole cycle (and prior cycles), writes report + dashboard."""

    name = "analysis"
    description = (
        "Make sense of this cycle. Read all sibling directories under the\n"
        "cycle root, plus prior cycles via `../../`. Write a human-facing\n"
        "report, dashboard, and (optional) a brief summary the next\n"
        "cycle's orchestrator will read first."
    )


def default_roles() -> list[_StubRole]:
    """Return the canonical 4+1 role lineup, in default execution order."""
    return [
        OrchestratorRole(),
        DataRole(),
        TrainingRole(),
        EvaluationRole(),
        AnalysisRole(),
    ]
