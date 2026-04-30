"""Deprecated location for ``agent_evolve.agents.mcp_mh``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.agents.mcp_mh`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.agents.mcp_mh import *  # noqa: F401,F403
