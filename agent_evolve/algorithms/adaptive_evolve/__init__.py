"""Deprecated location for ``agent_evolve.algorithms.adaptive_evolve``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.algorithms.adaptive_evolve`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.algorithms.adaptive_evolve import *  # noqa: F401,F403
