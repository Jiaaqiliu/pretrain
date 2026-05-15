"""Agent Teams adapter for nemo_mas — the only runtime.

Exposes the nemo_mas memory + backend surface over MCP so Claude Code
teammates (spawned via ``.claude/agents/nemo_mas_*.md``) can drive the
training loop conversationally. Shares ``memory``, ``schema``, and
``backends`` in the parent package.

Entry points:
  * ``server.main()``     — ``python -m agent_evolve.model.algorithms.nemo_mas.agent_teams.server``
  * ``hook_utils.*``      — helpers for ``.claude/hooks/nemo_mas_*.py``
  * ``role_guard.*``      — per-caller role validation for ``mem_write``
"""

from .hook_utils import (
    count_records_of_kind,
    current_memory_path,
    current_work_dir,
    current_workspace_root,
    cycle_workspace_path,
    meta_path,
    read_meta,
    read_records_jsonl,
)
from .role_guard import (
    VALID_WORKER_ROLES,
    RoleGuardError,
    check_worker_role,
)

__all__ = [
    "RoleGuardError",
    "VALID_WORKER_ROLES",
    "check_worker_role",
    "count_records_of_kind",
    "current_memory_path",
    "current_work_dir",
    "current_workspace_root",
    "cycle_workspace_path",
    "meta_path",
    "read_meta",
    "read_records_jsonl",
]
