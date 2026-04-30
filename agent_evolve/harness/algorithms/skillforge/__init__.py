"""SkillForge -- LLM-driven workspace mutation algorithm.

Canonical package name for the former ``aevolve`` implementation.
"""

from .engine import AEvolveEngine
from .prompts import DEFAULT_EVOLVER_SYSTEM_PROMPT
from .tools import BASH_TOOL_SPEC, make_workspace_bash

__all__ = [
    "AEvolveEngine",
    "DEFAULT_EVOLVER_SYSTEM_PROMPT",
    "BASH_TOOL_SPEC",
    "make_workspace_bash",
]
