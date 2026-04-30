"""AdaptiveSkill -- LLM-driven workspace mutation algorithm.

Trajectory-only evolution: an LLM analyzes agent trajectories, judges
performance, and mutates the agent workspace (prompts, skills, memory)
via bash tool access.
"""

from .engine import AdaptiveSkillEngine
from .prompts import DEFAULT_EVOLVER_SYSTEM_PROMPT
from .tools import BASH_TOOL_SPEC, make_workspace_bash

__all__ = [
    "AdaptiveSkillEngine",
    "DEFAULT_EVOLVER_SYSTEM_PROMPT",
    "BASH_TOOL_SPEC",
    "make_workspace_bash",
]
