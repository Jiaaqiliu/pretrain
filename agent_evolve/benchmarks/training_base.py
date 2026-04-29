"""Base protocol for training benchmarks.

Deliberately **not** a subclass of :class:`BenchmarkAdapter` because training
benchmarks evaluate checkpoints/adapters, not agent trajectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from ..training.data.base import (
    CoTRenderer,
    DataSynthGenerator,
    Solver,
    TrainingExample,
    Verifier,
)
from ..training.types import (
    CheckpointRef,
    ErrorBuckets,
    EvalMetrics,
    EvalPlan,
    MetricSpec,
    TrainingTrialResult,
    ValidityReport,
)


@runtime_checkable
class TrainingBenchmarkAdapter(Protocol):
    name: str

    def primary_metric(self) -> MetricSpec: ...

    def build_eval_plan(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        split: str,
    ) -> EvalPlan: ...

    def evaluate(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        backend: Any,
        split: str,
    ) -> Any: ...

    def parse_metrics(self, result_dir: Path) -> EvalMetrics: ...

    def analyze_errors(
        self,
        result_dir: Path,
        metrics: EvalMetrics,
    ) -> ErrorBuckets: ...

    def check_validity(
        self,
        workspace: Any,
        trial_result: TrainingTrialResult,
    ) -> ValidityReport: ...

    # ── Optional: data-pipeline hooks ─────────────────────────────────
    #
    # Benchmarks that want solver_distill / ood_augment stages implement
    # these; benchmarks without them keep working (teacher-only pipeline
    # is still supported via synth_generate). All four methods are
    # optional — presence detected at runtime in the stage workers.

    def iter_training_rows(
        self,
        workspace: Any,
    ) -> Iterable[TrainingExample]:
        """Yield ``TrainingExample`` objects over the benchmark's
        training corpus. Only consumed by data stages; unrelated to eval."""
        ...

    def classify_category(self, prompt: str) -> str:
        """Map a raw prompt string to a category key. Must be
        deterministic and cheap (regex / keyword match)."""
        ...

    def solvers(self) -> dict[str, Solver]:
        """Per-category deterministic solvers. Keys are category names
        (must match ``classify_category`` outputs). Empty dict means no
        solver support — ``solver_distill`` becomes a no-op for this
        benchmark."""
        ...

    def verifiers(self) -> dict[str, Verifier]:
        """Per-category answer verifiers. Both ``solver_distill`` and
        ``synth_generate`` (when ``verifier_gate: true``) consult these."""
        ...

    def cot_renderers(self) -> dict[str, CoTRenderer]:
        """Per-category CoT renderers. Required iff ``solvers()`` is
        non-empty — the stage worker checks."""
        ...

    def data_synth_generators(self) -> dict[str, DataSynthGenerator]:
        """Per-category OOD / synthetic puzzle generators. Phase 3 —
        consumed by a future ``ood_augment`` stage. Safe to leave empty."""
        ...


__all__ = ["TrainingBenchmarkAdapter"]
