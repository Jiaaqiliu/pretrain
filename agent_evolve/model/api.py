"""TrainingEvolver — public facade for training evolution.

Usage::

    import agent_evolve as ae

    evolver = ae.TrainingEvolver(
        workspace="seed_workspaces/nemotron_reasoner",
        benchmark="nemo_reasoner",
        algorithm="mcgs",
        backend="h200_single_node",
    )
    result = evolver.run(cycles=1)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from .registries import (
    resolve_algorithm,
    resolve_backend,
    resolve_benchmark,
)
from .types import (
    TrainingEvolutionResult,
    TrainingEvolveConfig,
    TrainingWorkspaceNotFound,
)
from .workspace import TrainingWorkspace

logger = logging.getLogger(__name__)


class TrainingEvolver:
    """Top-level API for training evolution."""

    def __init__(
        self,
        workspace: str | Path | TrainingWorkspace,
        benchmark: str | Any,
        algorithm: str | Any = "mcgs",
        backend: str | Any = "h200_single_node",
        config: TrainingEvolveConfig | str | Path | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.config = self._resolve_config(config)
        self.work_dir = Path(work_dir) if work_dir else Path("./training_evolution_workdir")
        self.workspace = self._resolve_workspace(workspace)
        self.benchmark = resolve_benchmark(benchmark)
        self.algorithm = resolve_algorithm(algorithm)
        self.backend = resolve_backend(backend)
        # Propagate the global smoke flag into backends that expose a ``mock``
        # toggle (e.g. SingleNodeTinkerLiteBackend). Without this, the CLI's
        # ``--smoke`` and the backend's ``mock`` drift apart.
        if hasattr(self.backend, "mock"):
            self.backend.mock = bool(self.config.smoke)
        self._loop = self._build_loop()

    def run(self, cycles: int | None = None) -> TrainingEvolutionResult:
        return self._loop.run(cycles=cycles)

    # ── resolution helpers ───────────────────────────────────────────

    @staticmethod
    def _resolve_config(
        config: TrainingEvolveConfig | str | Path | None,
    ) -> TrainingEvolveConfig:
        if config is None:
            return TrainingEvolveConfig()
        if isinstance(config, TrainingEvolveConfig):
            return config
        import yaml

        with open(config) as f:
            raw = yaml.safe_load(f) or {}
        return TrainingEvolveConfig(**raw)

    def _resolve_workspace(
        self, workspace: str | Path | TrainingWorkspace
    ) -> TrainingWorkspace:
        if isinstance(workspace, TrainingWorkspace):
            workspace.validate()
            return workspace
        src = Path(workspace)
        if not src.exists():
            raise TrainingWorkspaceNotFound(f"No training workspace at {src}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        dest = self.work_dir / src.name
        if not dest.exists():
            shutil.copytree(src, dest)
        return TrainingWorkspace.load(dest)

    def _build_loop(self) -> Any:
        # Defer to real loop if available (PR3+), otherwise use null loop.
        try:
            from .loop import TrainingEvolutionLoop

            return TrainingEvolutionLoop(
                workspace=self.workspace,
                benchmark=self.benchmark,
                algorithm=self.algorithm,
                backend=self.backend,
                config=self.config,
                work_dir=self.work_dir,
            )
        except ImportError:
            return _NullLoop(self.workspace, self.work_dir)


class _NullLoop:
    """Placeholder used before the training loop module is available.

    Returns an empty :class:`TrainingEvolutionResult`. Raises on attempts to
    run more than zero cycles so that downstream tests don't silently succeed.
    """

    def __init__(self, workspace: TrainingWorkspace, work_dir: Path):
        self._workspace = workspace
        self._work_dir = work_dir

    def run(self, cycles: int | None = None) -> TrainingEvolutionResult:
        if cycles and cycles > 0:
            raise RuntimeError(
                "TrainingEvolutionLoop is not installed yet. "
                "Install the training loop module to run cycles > 0."
            )
        return TrainingEvolutionResult(
            workspace=str(self._workspace.root),
            cycles_completed=0,
            incumbent_node_id=None,
            best_metric=None,
            best_checkpoint=None,
            graph_path=str(self._workspace.root / "evolution" / "mcgs_graph.json"),
            report_path=str(self._workspace.root / "evolution" / "reports"),
        )
