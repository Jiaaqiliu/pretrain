"""Data-generation pipeline primitives — benchmark-agnostic.

The training package owns the *shape* of data generation (Solver / Verifier /
CoTRenderer / DataSynthGenerator Protocols, recipe schema, stage workers,
dedup and CoT post-processing utilities).

Benchmarks (``agent_evolve/benchmarks/<name>/``) own the *substance* —
category-specific solvers, verifiers, and CoT templates. They wire
implementations to categories via a dict and expose the wiring through
``TrainingBenchmarkAdapter.solvers()`` / ``.verifiers()`` / ``.cot_renderers()``.

Stage types added by this package (dispatched in single_node._run_pipeline):

  - ``solver_distill``  — run each category's solver on train rows,
                          verify, render CoT, emit JSONL + stats.
  - ``data_merge``      — dedup across inputs, apply recipe upsample
                          ratios, append to ``data/sources.yaml``.

The existing ``synth_generate`` stage (teacher-LLM distillation) stays
unchanged except for an optional ``verifier_gate: true`` field that tells
it to apply the same correctness filter as solver_distill.
"""

from .base import (
    CoTRenderer,
    DataSynthGenerator,
    GeneratedRow,
    Solver,
    SolverResult,
    TrainingExample,
    Verifier,
)
from .generator import (
    DATA_GENERATORS,
    DataGenerator,
    register_data_generator,
    resolve_data_generator,
)
from .recipe import DataRecipe, load_recipe

# Import built-in generators for their @register_data_generator side effects.
from . import generators  # noqa: F401

__all__ = [
    "CoTRenderer",
    "DATA_GENERATORS",
    "DataGenerator",
    "DataRecipe",
    "DataSynthGenerator",
    "GeneratedRow",
    "Solver",
    "SolverResult",
    "TrainingExample",
    "Verifier",
    "load_recipe",
    "register_data_generator",
    "resolve_data_generator",
]
