"""Deprecated location for ``agent_evolve.agents``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.agents`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.agents import *  # noqa: F401,F403
