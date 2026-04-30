"""MasAdaptiveSkillEngine -- Multi-Agent System evolver.

Uses 4 specialized Strands agents (Orchestrator, Analyst, Author, Critic)
to decompose the evolution loop. Only the Orchestrator has tool access;
subagents are pure reasoning agents dispatched via @tool functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...config import EvolveConfig
from ...contract.workspace import AgentWorkspace
from ...engine.base import EvolutionEngine
from ...engine.versioning import VersionControl
from ...types import Observation, StepResult
from .orchestrator import run_evolution_cycle

logger = logging.getLogger(__name__)


class MasAdaptiveSkillEngine(EvolutionEngine):
    """Multi-agent workspace evolution engine.

    Decomposes the single-agent evolver into Orchestrator + Analyst +
    Author + Critic agents using the Strands @tool dispatch pattern.
    """

    def __init__(self, config: EvolveConfig):
        self.config = config

    def step(
        self,
        workspace: AgentWorkspace,
        observations: list[Observation],
        history: Any,
        trial: Any,
    ) -> StepResult:
        """Run one evolution step via multi-agent orchestration."""
        recent_logs = history.get_observations(last_n_cycles=2)
        cycle_num = history.latest_cycle + 1
        observation_dicts = self._observations_to_dicts(recent_logs)
        result = self._run_mas_cycle(workspace, observation_dicts, cycle_num)
        return StepResult(
            mutated=result.get("new_skills", 0) > 0,
            summary=f"MAS-Evolve: {result.get('new_skills', 0)} new skills",
            metadata=result,
        )

    def evolve(
        self,
        workspace: AgentWorkspace,
        observation_logs: list[dict[str, Any]],
        evo_number: int = 0,
    ) -> dict[str, Any]:
        """Run one evolution pass (called by batch_evolve_terminal.py).

        Signature matches AdaptiveSkillEngine.evolve() for drop-in use.
        """
        return self._run_mas_cycle(workspace, observation_logs, evo_number)

    def _run_mas_cycle(
        self,
        workspace: AgentWorkspace,
        observation_logs: list[dict[str, Any]],
        evo_number: int,
    ) -> dict[str, Any]:
        """Core evolution cycle shared by step() and evolve()."""
        vc = VersionControl(workspace.root)
        vc.init()

        skills_before = [s.name for s in workspace.list_skills()]

        vc.commit(
            message=f"pre-evo-{evo_number}: snapshot before MAS evolution",
            tag=f"pre-evo-{evo_number}",
        )

        region = self.config.extra.get("region", "us-west-2")
        mas_result = run_evolution_cycle(
            workspace_root=workspace.root,
            observation_logs=observation_logs,
            config_extra=self.config.extra,
            model_id=self.config.evolver_model,
            region=region,
            max_tokens=self.config.evolver_max_tokens,
        )

        # Programmatic size enforcement: reject oversized skills
        MAX_SKILL_CHARS = 2200
        skills_after_raw = workspace.list_skills()
        for skill in skills_after_raw:
            if skill.name not in skills_before:
                content = workspace.read_skill(skill.name)
                if content and len(content) > MAX_SKILL_CHARS:
                    logger.warning(
                        "Rejecting oversized skill '%s' (%d chars > %d limit)",
                        skill.name, len(content), MAX_SKILL_CHARS,
                    )
                    workspace.delete_skill(skill.name)

        skills_after = [s.name for s in workspace.list_skills()]
        new_skills = len(set(skills_after) - set(skills_before))
        added = sorted(set(skills_after) - set(skills_before))
        removed = sorted(set(skills_before) - set(skills_after))

        workspace.clear_drafts()

        mutated = set(skills_after) != set(skills_before)
        if mutated:
            vc.commit(
                message=f"evo-{evo_number}: MAS +{new_skills} skills {added}",
                tag=f"evo-{evo_number}",
            )
        else:
            vc.commit(
                message=f"evo-{evo_number}: MAS no mutation",
                tag=f"evo-{evo_number}",
            )

        logger.info(
            "MAS evo-%d: %d→%d skills (added=%s, removed=%s)",
            evo_number,
            len(skills_before),
            len(skills_after),
            added,
            removed,
        )

        return {
            "evo_number": evo_number,
            "tasks_analyzed": len(observation_logs),
            "skills_before": len(skills_before),
            "skills_after": len(skills_after),
            "new_skills": new_skills,
            "skills_added": added,
            "skills_removed": removed,
            "agent_calls": mas_result.get("agent_calls", {}),
            "total_steps": mas_result.get("total_steps", 0),
        }

    @staticmethod
    def _observations_to_dicts(observations: list) -> list[dict[str, Any]]:
        """Convert Observation objects or raw dicts to the dict format needed."""
        if not observations:
            return []
        if isinstance(observations[0], dict):
            return observations
        return [
            {
                "task_id": obs.task.id,
                "conversation": obs.trajectory.conversation,
            }
            for obs in observations
        ]
