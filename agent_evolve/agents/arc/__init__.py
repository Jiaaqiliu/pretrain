"""Deprecated location for ``agent_evolve.agents.arc``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.agents.arc`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.agents.arc import *  # noqa: F401,F403
