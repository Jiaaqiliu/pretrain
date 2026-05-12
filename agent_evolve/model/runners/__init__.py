"""Single-node training runners — shipped in smoke-friendly form for PR7.

Importing any symbol from this package pulls in the stage modules, which
register themselves with the ``stage_registry``. So importing
``agent_evolve.model.runners`` at all is sufficient to activate the
built-in stage types (``sft``, ``rl``, ``synth_generate``, ``solver_distill``,
``data_merge``, ``generate``).
"""

from .helpers.dataset import render_datums
from .helpers.pack_adapter import pack_adapter

# Import stage modules for their @register_stage side-effects. We also
# re-export a few entrypoints for backward compat with callers that
# reached in directly.
from .stages import generate as _stages_generate  # noqa: F401 — side-effect only
from .stages import sft_unsloth as _stages_sft_unsloth  # noqa: F401 — side-effect only
from .stages.data_merge import run_data_merge_stage  # noqa: F401
from .stages.eval import run_eval_plan
from .stages.rl import run_gspo_stage
from .stages.sft import run_sft_stage
from .stages.sft_unsloth import run_sft_unsloth_stage  # noqa: F401
from .stages.solver_distill import run_solver_distill_stage  # noqa: F401
from .stages.teacher_distill import run_synth_stage

__all__ = [
    "render_datums",
    "run_sft_stage",
    "run_synth_stage",
    "run_gspo_stage",
    "run_eval_plan",
    "pack_adapter",
]
