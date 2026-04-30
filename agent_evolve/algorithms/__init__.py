"""Deprecated location for ``agent_evolve.algorithms``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.algorithms`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.algorithms import *  # noqa: F401,F403
