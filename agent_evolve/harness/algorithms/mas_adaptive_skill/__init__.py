"""MAS Adaptive Skill -- Multi-Agent System evolver for skill evolution.

Decomposes the single-agent evolution loop into 4 specialized agents:
- Orchestrator: coordinates the cycle, has workspace_bash access
- Analyst: trajectory analysis, failure pattern identification
- Author: skill creation for identified patterns
- Critic: adversarial review of candidate skills

Uses Strands Agent + @tool dispatch pattern.
"""

from .engine import MasAdaptiveSkillEngine

__all__ = ["MasAdaptiveSkillEngine"]
