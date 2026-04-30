"""Built-in ``DataGenerator`` implementations.

Each module here wraps an existing stage worker
(``runners/stages/solver_distill.py``, ``teacher_distill.py``) in a
``DataGenerator`` class so the new ``type: generate, generator: <name>``
pipeline syntax can dispatch to it. The old stage types
(``type: solver_distill`` / ``type: synth_generate``) keep working via the
original stage-registry entries — see ``runners/stages/*``.

Importing this subpackage triggers ``@register_data_generator`` side
effects for all built-in generators.
"""

from . import solver_distill  # noqa: F401
from . import teacher_llm  # noqa: F401

__all__: list[str] = []
