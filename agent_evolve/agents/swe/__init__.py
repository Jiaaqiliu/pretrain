"""Deprecated location for ``agent_evolve.agents.swe``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.agents.swe`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.agents.swe import *  # noqa: F401,F403
