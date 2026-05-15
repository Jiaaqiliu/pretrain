"""Per-caller role guard for MCP tool invocations.

MCP has no native concept of "which teammate called this tool" — each
call arrives over the same stdio pipe regardless of subagent. To preserve
the per-role ``KIND_WHITELIST`` enforcement that the in-process Bedrock
runtime gets for free via ``SpawnHandler``, MCP tools require an explicit
``role`` argument. The guard rejects unknown or orchestrator-reserved
roles on ``mem_write``.

This is best-effort. A malicious teammate could lie about its role; the
guard defends against *accidental* role confusion, not an adversarial
subagent. If that becomes a concern we can bind role to MCP connection
metadata once Claude Code surfaces it.
"""

from __future__ import annotations

from ..schema import KIND_WHITELIST


# Roles a teammate is allowed to declare on MCP calls. Orchestrator is
# absent because in Agent Teams the lead plays the orchestrator role
# directly; no separate worker needs to call ``mem_write`` as the
# orchestrator.
VALID_WORKER_ROLES = frozenset({"planner", "data_worker", "trainer"})


class RoleGuardError(ValueError):
    """Raised when a caller declares a role that isn't allowed to invoke
    the tool they're calling. Surfaced to the LLM as a tool error string."""


def check_worker_role(role: str) -> None:
    """Tool-call precondition for ``mem_write`` and data-writing tools.

    Raises ``RoleGuardError`` if ``role`` is outside the worker whitelist.
    """
    if role not in VALID_WORKER_ROLES:
        raise RoleGuardError(
            f"role={role!r} is not a valid worker role for mem_write; "
            f"expected one of {sorted(VALID_WORKER_ROLES)}."
        )
    # Extra belt-and-suspenders: the role must also exist in the
    # schema's whitelist (catches typos that sneak past VALID_WORKER_ROLES).
    if role not in KIND_WHITELIST:
        raise RoleGuardError(
            f"role={role!r} has no entry in KIND_WHITELIST — refusing mem_write"
        )
