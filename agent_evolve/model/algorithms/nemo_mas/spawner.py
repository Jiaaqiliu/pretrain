"""SpawnHandler — turns orchestrator's spawn requests into BedrockAgent runs.

Mirrors arc-mas's ``spawn_and_run_subagent`` (mas_agent.py:406-466) with
two differences:

  1. Workers don't share a frame / game-state object — they share the
     ``RecipeMemory`` instance, which is the only inter-worker channel.
  2. The orchestrator's ``task`` message is the worker's first user
     message (per the design — "first-round user query is provided by
     the orchestrator").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable

from .memory import RecipeMemory
from .schema import KIND_WHITELIST, kinds_for_role
from .tools import BackendToolRegistry, build_role_tools

logger = logging.getLogger(__name__)


# Lazy import — BedrockAgent has heavy boto3 import side-effects we want
# to avoid at module load time (e.g., during tests of memory.py).
def _import_bedrock_agent():
    from agent_evolve.harness.agents.arc.bedrock_agent import BedrockAgent
    return BedrockAgent


# ── Per-role system prompt assembly ─────────────────────────────────


def _memory_protocol_block(role: str) -> str:
    allowed = ", ".join(kinds_for_role(role))
    return f"""
# Runtime memory protocol (system-injected)

You can write the following record kinds via mem_write: {allowed}.
Out-of-whitelist kinds are rejected with an error. Per-kind ref rules
are enforced (see kind=breakthrough requires refs; recipe_proposal
requires ref to eval_report or data_gap; training_run requires refs to
recipe_proposal AND dataset_snapshot; cv_result requires ref to
training_run; eval_report requires ref to training_run).

Always start by calling mem_recent(kind="breakthrough") to load the
global priors before doing anything else.
""".strip()


def _skill_protocol_block(role: str, suggested: Iterable[str] | None) -> str:
    suggestions = list(suggested or [])
    sugg = (
        f"\nSuggested skills for this task: {suggestions}\n"
        "Load them first via skill_load — they are calibrated to your task."
        if suggestions else
        "\nUse skill_index() first to discover available skills, then "
        "skill_load(name) to read one in full."
    )
    return f"""
# Runtime skill protocol (system-injected)

Skills live under skills/{role}/. Prefer loading and following a skill's
procedure over reasoning from scratch when a skill matches your task.
{sugg}
""".strip()


def _build_system_prompt(
    *,
    role: str,
    base_prompt: str,
    benchmark_reference: str,
    breakthroughs_block: str,
    suggested_skills: Iterable[str] | None,
) -> str:
    parts = [base_prompt.strip()]
    if benchmark_reference.strip():
        parts.append(benchmark_reference.strip())
    if breakthroughs_block.strip():
        parts.append(breakthroughs_block.strip())
    parts.append(_memory_protocol_block(role))
    parts.append(_skill_protocol_block(role, suggested_skills))
    return "\n\n---\n\n".join(parts)


# ── SpawnHandler ────────────────────────────────────────────────────


class SpawnHandler:
    """Spawns + tracks worker agents.

    One instance per cycle. The orchestrator binds two of its tools to
    methods here: ``spawn_and_run_subagent`` and ``call_existing_agent``.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        memory: RecipeMemory,
        backend_registry: BackendToolRegistry | None = None,
        model_id: str = "us.anthropic.claude-opus-4-6-v1",
        thinking_effort: str = "",
        max_tokens_default: int = 16384,
        max_tokens_thinking: int = 65536,
    ):
        self.workspace_root = Path(workspace_root)
        self.memory = memory
        self.backend_registry = backend_registry
        self.model_id = model_id
        self.thinking_effort = thinking_effort
        self.max_tokens_default = max_tokens_default
        self.max_tokens_thinking = max_tokens_thinking

        self._agents: dict[str, object] = {}      # agent_id -> BedrockAgent
        self._counter: int = 0

        prompts_dir = self.workspace_root / "prompts"
        skills_root = self.workspace_root / "skills"
        self._prompts_dir = prompts_dir
        self._skills_root = skills_root

        # Cache base prompts + benchmark reference (read once per cycle).
        self._base_prompts: dict[str, str] = {}
        for role in KIND_WHITELIST:
            p = prompts_dir / f"{role}.md"
            self._base_prompts[role] = p.read_text(encoding="utf-8") if p.exists() else ""
        ref = prompts_dir / "benchmark_reference.md"
        self._benchmark_reference = ref.read_text(encoding="utf-8") if ref.exists() else ""

    # ── Public methods (tool handlers) ─────────────────────────

    def spawn_and_run_subagent(
        self,
        *,
        role: str,
        task: str,
        suggested_skills: list[str] | None = None,
        budget_tokens: int | None = None,
    ) -> str:
        """Tool handler for orchestrator → spawn one focused task."""
        if role not in KIND_WHITELIST:
            return json.dumps({
                "ok": False,
                "error": f"unknown role {role!r}; expected one of {sorted(KIND_WHITELIST)}",
            })

        before_ids = {r.id for r in self.memory.all_records()}

        BedrockAgent = _import_bedrock_agent()

        self._counter += 1
        agent_id = f"{role}_{self._counter}"

        breakthroughs = self.memory.breakthroughs_block()
        system_prompt = _build_system_prompt(
            role=role,
            base_prompt=self._base_prompts.get(role, ""),
            benchmark_reference=self._benchmark_reference,
            breakthroughs_block=breakthroughs,
            suggested_skills=suggested_skills,
        )

        specs, handlers = build_role_tools(
            role,
            memory=self.memory,
            skills_root=self._skills_root,
            workspace_root=self.workspace_root,
            backend_registry=self.backend_registry,
        )

        max_tokens = (
            self.max_tokens_thinking if self.thinking_effort else self.max_tokens_default
        )

        try:
            agent = BedrockAgent(
                model_id=self.model_id,
                system_prompt=system_prompt,
                tools=specs,
                tool_handlers=handlers,
                agent_id=self._counter,
                max_tokens=max_tokens,
                thinking_effort=self.thinking_effort,
            )
        except Exception as e:           # noqa: BLE001 — boto/import surface
            logger.exception("Failed to construct BedrockAgent for %s", agent_id)
            return json.dumps({"ok": False, "error": f"agent construction failed: {e}"})

        self._agents[agent_id] = agent

        try:
            result_text = agent.call(task)
        except Exception as e:           # noqa: BLE001
            logger.exception("Agent %s.call raised", agent_id)
            return json.dumps({"ok": False, "agent_id": agent_id,
                               "error": f"agent.call raised: {e}"})

        new_ids = [r.id for r in self.memory.all_records() if r.id not in before_ids]
        return json.dumps({
            "ok": True,
            "agent_id": agent_id,
            "result": result_text,
            "new_record_ids": new_ids,
        })

    def call_existing_agent(self, *, agent_id: str, task: str) -> str:
        """Tool handler for orchestrator → resume a prior worker."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return json.dumps({"ok": False,
                               "error": f"agent {agent_id!r} not found in this cycle"})
        before_ids = {r.id for r in self.memory.all_records()}
        try:
            result_text = agent.call(task)  # type: ignore[attr-defined]
        except Exception as e:               # noqa: BLE001
            logger.exception("Agent %s.call raised", agent_id)
            return json.dumps({"ok": False, "agent_id": agent_id,
                               "error": f"agent.call raised: {e}"})
        new_ids = [r.id for r in self.memory.all_records() if r.id not in before_ids]
        return json.dumps({"ok": True, "agent_id": agent_id,
                           "result": result_text, "new_record_ids": new_ids})

    # ── Handler helpers (specs come from YAML; orchestrator.py merges these in) ─

    @property
    def spawn_specs_and_handlers(self) -> tuple[list[dict], dict[str, Callable[..., str]]]:
        """Return ``(specs=[], handlers={...})`` for spawn tools.

        Specs come from ``_common_model/tools/orchestrator.yaml`` now;
        this method kept for backward compatibility returns an empty
        spec list and the two handler callables.
        """
        handlers: dict[str, Callable[..., str]] = {
            "spawn_and_run_subagent": self.spawn_and_run_subagent,
            "call_existing_agent": self.call_existing_agent,
        }
        return [], handlers
