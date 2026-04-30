"""Deprecated location for ``agent_evolve.algorithms.gepa``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.algorithms.gepa`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.algorithms.gepa import *  # noqa: F401,F403
