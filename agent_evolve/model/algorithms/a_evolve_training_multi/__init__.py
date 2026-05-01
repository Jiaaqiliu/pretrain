"""a-evolve-training-multi — minimal multi-role evolution algorithm.

The contract is intentionally tiny. The platform fixes:

1. **Cycle layout**: each cycle is a directory under
   ``<workspace>/cycles/<NNNN>/``.
2. **Role slots**: each role gets a subdirectory named after it; the
   platform creates the dir and a ``_done`` sentinel when the cycle
   finishes.
3. **Five role names + default execution order**:
   ``orchestrator → data → training → evaluation → analysis``.

The platform does **not** fix:

* What files a role writes (md / yaml / jsonl / png / ... — pick one).
* What schemas those files use.
* How roles communicate (read sibling dirs by convention).
* How a role is implemented (LLM, MCGS, Claude Code, plain Python,
  nested multi-agent — all fine, all replaceable).

See ``role.py`` (5-line ``Role`` Protocol) and ``loop.py``
(~20-line ``run_cycle``). Everything else is replaceable per-team.
"""
from __future__ import annotations

from .loop import next_cycle_id, run_cycle
from .role import Role
from .roles import (
    AnalysisRole,
    DataRole,
    EvaluationRole,
    OrchestratorRole,
    TrainingRole,
    default_roles,
)

__all__ = [
    "Role",
    "run_cycle",
    "next_cycle_id",
    "OrchestratorRole",
    "DataRole",
    "TrainingRole",
    "EvaluationRole",
    "AnalysisRole",
    "default_roles",
]
