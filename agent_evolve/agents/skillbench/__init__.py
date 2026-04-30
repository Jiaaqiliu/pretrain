"""Deprecated location for ``agent_evolve.agents.skillbench``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.agents.skillbench`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.agents.skillbench import *  # noqa: F401,F403
