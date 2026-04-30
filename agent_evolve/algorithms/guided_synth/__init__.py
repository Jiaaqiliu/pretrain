"""Deprecated location for ``agent_evolve.algorithms.guided_synth``.

This is a thin re-export shim. New code should import from
``agent_evolve.harness.algorithms.guided_synth`` directly. The shim package keeps
existing third-party callers working through the deprecated path.
"""

from agent_evolve.harness.algorithms.guided_synth import *  # noqa: F401,F403
