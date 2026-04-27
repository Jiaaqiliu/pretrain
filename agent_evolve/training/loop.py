"""TrainingEvolutionLoop — orchestrates MCGS cycles over a TrainingWorkspace.

Contract:
  - Loop calls ``algorithm.run_cycle(ctx)`` exactly once per cycle.
  - Loop persists each :class:`MCGSCycleReport` as JSON.
  - Loop materializes the incumbent only when ``report.incumbent_changed`` is
    True. No accept/reject gate lives here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .observer import TrainingObserver
from .trial import TrialRunner
from .types import (
    MCGSCycleReport,
    TrainingEvolutionResult,
    TrainingEvolveConfig,
    TrialBudget,
)

logger = logging.getLogger(__name__)


@dataclass
class LoopContext:
    """Per-cycle context passed to ``algorithm.run_cycle``."""

    cycle: int
    workspace: Any
    benchmark: Any
    backend: Any
    config: TrainingEvolveConfig
    work_dir: Path
    trial: TrialRunner
    observer: TrainingObserver
    budget: TrialBudget


class TrainingEvolutionLoop:
    def __init__(
        self,
        workspace: Any,
        benchmark: Any,
        algorithm: Any,
        backend: Any,
        config: TrainingEvolveConfig | None = None,
        work_dir: Path | None = None,
    ):
        self.workspace = workspace
        self.benchmark = benchmark
        self.algorithm = algorithm
        self.backend = backend
        self.config = config or TrainingEvolveConfig()
        self.work_dir = Path(work_dir) if work_dir else Path("./training_evolution_workdir")

        evolution_dir = Path(workspace.root) / "evolution"
        evolution_dir.mkdir(parents=True, exist_ok=True)
        self.observer = TrainingObserver(evolution_dir)
        self.trial = TrialRunner(backend, benchmark)

    def run(self, cycles: int | None = None) -> TrainingEvolutionResult:
        max_cycles = cycles if cycles is not None else self.config.max_cycles
        last_report: MCGSCycleReport | None = None
        incumbent_node = None
        best_metric: float | None = None
        best_checkpoint = None

        for cycle in range(1, max_cycles + 1):
            ctx = LoopContext(
                cycle=cycle,
                workspace=self.workspace,
                benchmark=self.benchmark,
                backend=self.backend,
                config=self.config,
                work_dir=self.work_dir,
                trial=self.trial,
                observer=self.observer,
                budget=TrialBudget(
                    seconds=self.config.trial_budget_seconds,
                    steps=self.config.trial_budget_steps,
                    tokens=self.config.trial_budget_tokens,
                ),
            )
            logger.info("=== TrainingEvolution cycle %d/%d ===", cycle, max_cycles)
            report = self.algorithm.run_cycle(ctx)
            report_path = self.observer.record_cycle(report)
            # Record the persistent path so downstream consumers can find it.
            try:
                report.report_path = str(report_path)
            except AttributeError:  # pragma: no cover — dataclass is mutable
                pass
            last_report = report

            if report.incumbent_changed and report.incumbent_node_id is not None:
                incumbent_node = _lookup_incumbent_node(self.algorithm, report.incumbent_node_id)
                if incumbent_node is not None:
                    self.workspace.materialize_incumbent(incumbent_node)
                    best_metric = incumbent_node.metric
                    best_checkpoint = incumbent_node.checkpoint

        graph_path = _maybe_graph_path(self.algorithm, self.workspace)
        report_path = (
            last_report.report_path
            if last_report and getattr(last_report, "report_path", None)
            else str(Path(self.workspace.root) / "evolution" / "reports")
        )
        topk = []
        if hasattr(self.algorithm, "topk_summary"):
            try:
                topk = self.algorithm.topk_summary()
            except Exception:  # pragma: no cover
                topk = []

        return TrainingEvolutionResult(
            workspace=str(self.workspace.root),
            cycles_completed=max_cycles,
            incumbent_node_id=last_report.incumbent_node_id if last_report else None,
            best_metric=best_metric,
            best_checkpoint=best_checkpoint,
            graph_path=graph_path,
            report_path=report_path,
            topk=topk,
        )


# ── Helpers ──────────────────────────────────────────────────────────────

def _lookup_incumbent_node(algorithm: Any, node_id: str):
    """Retrieve a node from the algorithm's graph by id, if available."""
    graph = getattr(algorithm, "graph", None)
    if graph is None:
        return None
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        return nodes.get(node_id)
    if hasattr(graph, "get_node"):
        try:
            return graph.get_node(node_id)
        except Exception:  # pragma: no cover
            return None
    return None


def _maybe_graph_path(algorithm: Any, workspace: Any) -> str:
    path = getattr(algorithm, "graph_path", None)
    if path:
        return str(path)
    return str(Path(workspace.root) / "evolution" / "mcgs_graph.json")
