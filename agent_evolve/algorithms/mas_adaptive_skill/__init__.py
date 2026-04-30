"""Deprecated location for ``agent_evolve.algorithms.mas_adaptive_skill``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.algorithms.mas_adaptive_skill`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.algorithms.mas_adaptive_skill import *  # noqa: F401,F403
