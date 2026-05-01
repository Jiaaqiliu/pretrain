"""Null object for the training search-algorithm Protocol.

Used by loop/backend integration tests to exercise infrastructure
(cycle counting, report persistence, workspace fork plumbing) without
dragging in MCGS's selection/mutation/reward logic. If a loop-level
invariant breaks, the NullSearchAlgorithm-based tests fail first —
cleanly separating "loop bug" from "algorithm bug".

Also doubles as the minimum viable reference implementation for anyone
adding a new search algorithm: the ~20 lines below show the entire
Protocol surface (``run_cycle(ctx) -> MCGSCycleReport``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import MCGSCycleReport


@dataclass
class NullSearchAlgorithm:
    """No-op search algorithm. Increments a counter per cycle, returns
    an empty ``MCGSCycleReport`` (no parent, no trials, no incumbent).
    Never forks the workspace, never calls the backend. Intended purely
    for scaffolding / infrastructure tests."""

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
