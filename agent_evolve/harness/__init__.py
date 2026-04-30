"""Harness / agent-evolve subsystem.

Top-level home for the agent side of agent_evolve:

  * ``protocol/``   — :class:`BaseAgent` and other agent-side Protocols
  * ``contract/``   — :class:`AgentWorkspace`, :class:`Manifest`, schema validation
  * ``engine/``     — evolution loop, trial runner, observer, history, versioning
  * ``algorithms/`` — SkillForge, GEPA, MetaHarness, AdaptiveEvolve, MasAdaptiveSkill, ...
  * ``agents/``     — ARC, MCP, SkillBench, SWE, Terminal, ...
  * ``api.py``      — :class:`Evolver`, the public top-level entry point

The model-training subsystem lives at ``agent_evolve.training``. Both
subsystems share infrastructure under ``agent_evolve.{benchmarks, backends,
llm, utils, types, config}``.
"""
