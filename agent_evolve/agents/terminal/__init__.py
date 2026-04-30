"""Deprecated location for ``agent_evolve.agents.terminal``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.agents.terminal`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.agents.terminal import *  # noqa: F401,F403
