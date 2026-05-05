"""Per-caller role guard for MCP tool invocations.

MCP has no native concept of "which teammate called this tool" — each
call arrives over the same stdio pipe regardless of subagent. To preserve
the per-role ``KIND_WHITELIST`` enforcement that the in-process Bedrock
runtime gets for free via ``SpawnHandler``, MCP tools require an explicit
``role`` argument.

Each subagent's system prompt (``.claude/agents/nemo_mas_*.md``) tells
the teammate to always pass ``role="<its-role>"`` on tool calls. The
guard:

  * rejects unknown or orchestrator-reserved roles on ``mem_write``,
  * maps the reviewer to the orchestrator-auto signer in auto mode when
    calling ``checkpoint_sign`` (same posture as today's
    ``_build_checkpoint_sign_handler(signer_role="reviewer")``),
  * allows the lead (``role="human"``) to sign in both modes — the
    lead IS the human in Agent Teams.

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
VALID_WORKER_ROLES = frozenset({"planner", "data_worker", "trainer", "reviewer"})

# Special caller identities outside the worker whitelist.
ROLE_HUMAN = "human"          # the lead (you), signing manually
ROLE_ORCHESTRATOR_AUTO = "orchestrator_auto"  # auto-mode signoff by the lead


class RoleGuardError(ValueError):
    """Raised when a caller declares a role that isn't allowed to invoke
    the tool they're calling. Surfaced to the LLM as a tool error string."""


def check_worker_role(role: str) -> None:
    """Tool-call precondition for ``mem_write`` and data-writing tools.

    Raises ``RoleGuardError`` if ``role`` is outside the worker whitelist.
    The orchestrator-auto / human roles write checkpoint events via
    dedicated tools (``checkpoint_sign``), not ``mem_write``, so they
    are intentionally rejected here.
    """
    if role not in VALID_WORKER_ROLES:
        raise RoleGuardError(
            f"role={role!r} is not a valid worker role for mem_write; "
            f"expected one of {sorted(VALID_WORKER_ROLES)}. "
            "Human sign-offs go through ``checkpoint_sign``; the lead's "
            "orchestrator reads memory via mem_* but does not write."
        )
    # Extra belt-and-suspenders: the role must also exist in the
    # schema's whitelist (catches typos that sneak past VALID_WORKER_ROLES).
    if role not in KIND_WHITELIST:
        raise RoleGuardError(
            f"role={role!r} has no entry in KIND_WHITELIST — refusing mem_write"
        )


def resolve_signer_role(declared_role: str) -> tuple[str, str]:
    """Map the caller's declared role to ``(signer_role, actor_label)``.

    ``signer_role`` is what ``memory.write`` validates against (must be
    a key in ``KIND_WHITELIST``). ``actor_label`` is the human-readable
    tag that lands on the ``checkpoint_event`` (``actor:<label>``).

    Mapping:
      * ``"human"``              → orchestrator_auto / ``human:lead``
      * ``"orchestrator_auto"``  → orchestrator_auto / ``orchestrator``
                                   (lead acting on the orchestrator's
                                   behalf in auto mode)
      * ``"reviewer"``           → reviewer / ``reviewer``

    Any other role is rejected — signing is NOT a general capability,
    only these three identities may produce a ``checkpoint_event``.
    """
    if declared_role == ROLE_HUMAN:
        return (ROLE_ORCHESTRATOR_AUTO, "human:lead")
    if declared_role == ROLE_ORCHESTRATOR_AUTO:
        return (ROLE_ORCHESTRATOR_AUTO, "orchestrator")
    if declared_role == "reviewer":
        return ("reviewer", "reviewer")
    raise RoleGuardError(
        f"role={declared_role!r} is not allowed to sign checkpoints; "
        f"only {sorted({ROLE_HUMAN, ROLE_ORCHESTRATOR_AUTO, 'reviewer'})}."
    )
