"""Deprecated location for ``agent_evolve.agents.skillbench.official_terminus``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.agents.skillbench.official_terminus`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.agents.skillbench.official_terminus import *  # noqa: F401,F403
