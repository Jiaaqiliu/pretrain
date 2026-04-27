"""Null search algorithm — a no-op used by PR3 tests before MCGS lands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import MCGSCycleReport


@dataclass
class NullSearchAlgorithm:
    """Simple algorithm for scaffolding tests. Does nothing on each cycle."""

    cycle: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def run_cycle(self, ctx: Any) -> MCGSCycleReport:
        self.cycle += 1
        self.calls.append({"cycle": self.cycle})
        return MCGSCycleReport(
            cycle=self.cycle,
            selected_parent_id=None,
            trial_node_ids=[],
            incumbent_node_id=None,
            incumbent_changed=False,
            best_metric=None,
            graph_path="",
            report_path="",
        )
