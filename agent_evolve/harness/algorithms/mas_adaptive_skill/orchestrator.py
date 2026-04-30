"""Multi-agent orchestrator for skill evolution.

Uses Strands Agent + @tool dispatch pattern:
- Orchestrator agent has workspace_bash + 3 subagent tools
- Subagents (Analyst, Author, Critic) are pure reasoning agents (no tools)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from strands import Agent, tool
from strands.hooks.events import BeforeToolCallEvent
from strands.models import BedrockModel

from ..adaptive_skill.tools import make_workspace_bash
from .prompts import (
    ANALYST_SYSTEM_PROMPT,
    AUTHOR_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    build_batch_data,
)

logger = logging.getLogger(__name__)

MAX_ORCHESTRATOR_STEPS = 50


def run_evolution_cycle(
    workspace_root: Path,
    observation_logs: list[dict[str, Any]],
    config_extra: dict[str, Any],
    model_id: str = "us.anthropic.claude-opus-4-6-v1",
    region: str = "us-west-2",
    max_tokens: int = 16384,
) -> dict[str, Any]:
    """Run one multi-agent evolution cycle.

    Args:
        workspace_root: Path to the agent workspace directory.
        observation_logs: List of observation dicts (task_id, conversation, ...).
        config_extra: Extra config from EvolveConfig.extra.
        model_id: Bedrock model ID for all agents.
        region: AWS region.
        max_tokens: Max tokens per agent response.

    Returns:
        Dict with agent_calls counts and usage info.
    """
    max_skills = config_extra.get("max_skills", 5)
    protect_skills = config_extra.get("protect_skills", True)

    # Read current workspace state for batch data
    from ...contract.workspace import AgentWorkspace

    ws = AgentWorkspace(workspace_root)
    existing_skills = ws.list_skills()
    existing_skill_names = [s.name for s in existing_skills]
    current_skill_count = len(existing_skills)

    # Read existing skill content for author context
    existing_skill_contents: dict[str, str] = {}
    for skill in existing_skills:
        content = ws.read_skill(skill.name)
        if content:
            existing_skill_contents[skill.name] = content

    # Look up task descriptions from observation logs
    task_descriptions: dict[str, str] = {}
    for log in observation_logs:
        task_id = log.get("task_id", "")
        desc = log.get("task_input", "")
        if task_id and desc:
            task_descriptions[task_id] = desc

    # Build batch data
    orchestrator_prompt, analyst_input = build_batch_data(
        observation_logs,
        max_skills=max_skills,
        current_skill_count=current_skill_count,
        existing_skill_names=existing_skill_names,
        existing_skill_contents=existing_skill_contents,
        protect_skills=protect_skills,
        task_descriptions=task_descriptions,
    )

    # Shared model for all agents
    model = BedrockModel(
        model_id=model_id,
        region_name=region,
        max_tokens=max_tokens,
    )

    # Track subagent call counts
    call_counts: dict[str, int] = {
        "analyze": 0,
        "author_skill": 0,
        "critique_skill": 0,
        "workspace_bash": 0,
    }

    # ── Subagent tools ──────────────────────────────────────────

    bash_fn = make_workspace_bash(workspace_root)

    @tool
    def workspace_bash(command: str) -> str:
        """Execute a bash command in the agent workspace directory. \
Use to read/write skill files and inspect git state."""
        call_counts["workspace_bash"] += 1
        return bash_fn(command)

    @tool
    def analyze(batch_trajectories: str) -> str:
        """Send trajectory data to the Analyst agent for failure pattern analysis. \
Returns a JSON array of failure patterns. Pass the full batch trajectory JSON."""
        call_counts["analyze"] += 1
        try:
            analyst = Agent(
                model=model,
                system_prompt=ANALYST_SYSTEM_PROMPT,
                tools=[],
            )
            result = analyst(batch_trajectories)
            return str(result)
        except Exception as e:
            logger.error("Analyst agent failed: %s", str(e)[:200])
            return f"ERROR: Analyst failed — {str(e)[:200]}"

    @tool
    def author_skill(pattern_and_context: str) -> str:
        """Send a failure pattern to the Author agent to create a candidate SKILL.md. \
Include the pattern details and existing skill names."""
        call_counts["author_skill"] += 1
        try:
            author = Agent(
                model=model,
                system_prompt=AUTHOR_SYSTEM_PROMPT,
                tools=[],
            )
            result = author(pattern_and_context)
            return str(result)
        except Exception as e:
            logger.error("Author agent failed: %s", str(e)[:200])
            return f"ERROR: Author failed — {str(e)[:200]}"

    @tool
    def critique_skill(skill_and_context: str) -> str:
        """Send a candidate skill to the Critic agent for adversarial review. \
Include the candidate SKILL.md content, the failure pattern, and existing skill names."""
        call_counts["critique_skill"] += 1
        try:
            critic = Agent(
                model=model,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                tools=[],
            )
            result = critic(skill_and_context)
            return str(result)
        except Exception as e:
            logger.error("Critic agent failed: %s", str(e)[:200])
            return f"ERROR: Critic failed — {str(e)[:200]}"

    # ── Orchestrator agent ──────────────────────────────────────

    orchestrator = Agent(
        model=model,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=[analyze, author_skill, critique_skill, workspace_bash],
    )

    # Step limiter to prevent runaway
    step_count = [0]

    def _step_limiter(event: BeforeToolCallEvent):
        step_count[0] += 1
        if step_count[0] > MAX_ORCHESTRATOR_STEPS:
            event.cancel_tool = (
                f"Step limit reached ({MAX_ORCHESTRATOR_STEPS}). "
                "Finish by verifying with `workspace_bash('git diff')` and summarizing."
            )

    orchestrator.hooks.add_callback(BeforeToolCallEvent, _step_limiter)

    # ── Run ─────────────────────────────────────────────────────

    logger.info(
        "MAS evolution: %d tasks, %d/%d skills, model=%s",
        len(observation_logs),
        current_skill_count,
        max_skills,
        model_id,
    )

    try:
        response = orchestrator(orchestrator_prompt)
        response_text = str(response)
    except Exception as e:
        logger.error("Orchestrator agent failed: %s", str(e)[:500])
        response_text = f"ERROR: {e}"

    logger.info(
        "MAS evolution complete: analyze=%d, author=%d, critic=%d, bash=%d",
        call_counts["analyze"],
        call_counts["author_skill"],
        call_counts["critique_skill"],
        call_counts["workspace_bash"],
    )

    return {
        "agent_calls": call_counts,
        "total_steps": step_count[0],
        "response_summary": response_text[:500],
    }
