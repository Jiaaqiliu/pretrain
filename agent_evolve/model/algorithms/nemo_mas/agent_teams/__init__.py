"""Agent Teams adapter for nemo_mas.

Interactive front-end — exposes the nemo_mas memory / checkpoint /
backend surface over MCP so Claude Code teammates (spawned via
``.claude/agents/nemo_mas_*.md``) can drive the Quality-Plan loop
conversationally. The headless ``run_cycle`` path in ``orchestrator.py``
is unaffected; both runtimes share ``memory``, ``schema``,
``checkpoints``, and ``backends`` in the parent package.

Entry points:
  * ``server.main()``     — ``python -m agent_evolve.model.algorithms.nemo_mas.agent_teams.server``
  * ``hook_utils.*``      — helpers for ``.claude/hooks/nemo_mas_*.py``
  * ``role_guard.*``      — per-caller role validation for ``mem_write``
                            and ``checkpoint_sign``
"""

from .hook_utils import (
    count_records_of_kind,
    current_checkpoint_mode,
    current_memory_path,
    current_work_dir,
    current_workspace_root,
    first_blocker_or_none,
    fold_current_run,
    format_blocker_message,
    meta_path,
    read_meta,
    read_records_jsonl,
)
from .role_guard import (
    ROLE_HUMAN,
    ROLE_ORCHESTRATOR_AUTO,
    VALID_WORKER_ROLES,
    RoleGuardError,
    check_worker_role,
    resolve_signer_role,
)

__all__ = [
    "ROLE_HUMAN",
    "ROLE_ORCHESTRATOR_AUTO",
    "RoleGuardError",
    "VALID_WORKER_ROLES",
    "check_worker_role",
    "count_records_of_kind",
    "current_checkpoint_mode",
    "current_memory_path",
    "current_work_dir",
    "current_workspace_root",
    "first_blocker_or_none",
    "fold_current_run",
    "format_blocker_message",
    "meta_path",
    "read_meta",
    "read_records_jsonl",
    "resolve_signer_role",
]
