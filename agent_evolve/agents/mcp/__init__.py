"""Deprecated location for ``agent_evolve.agents.mcp``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.agents.mcp`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.agents.mcp import *  # noqa: F401,F403
