"""Protocols and dataclasses for the data-generation pipeline.

These types are the *contract* every benchmark must satisfy to plug into
``solver_distill`` / ``data_merge`` stages. A benchmark does not have to
implement all of them — if no ``Solver`` is registered for a category,
that category is skipped in ``solver_distill``; teacher-only (``synth_generate``)
still works.

Design notes:
  * All four Protocols are ``runtime_checkable`` so tests can assert
    shape conformance on fakes without instantiating real benchmarks.
  * ``SolverResult`` and ``GeneratedRow`` are plain dataclasses with
    ``.to_dict()`` helpers so they serialize cleanly to JSONL without
    pulling pickle. The JSONL schema is the inter-stage wire format.
  * ``CoTRenderer.render`` takes a ``trace_dict`` on purpose — the
    solver decides what intermediate state to expose; the renderer is a
    pure function of (prompt, answer, trace). No LLM required for
    baseline renderers; a benchmark that wants a teacher-rewritten
    renderer can still implement the Protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Protocol, runtime_checkable


# ── Results ──────────────────────────────────────────────────────────────

Confidence = Literal["high", "low", "none"]


@dataclass
class TrainingExample:
    """A single (prompt, answer, category) row from a benchmark's training
    corpus. Benchmarks yield these via
    ``TrainingBenchmarkAdapter.iter_training_rows(workspace)``.
    """
    id: str
    prompt: str
    answer: str
    category: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Output of a ``Solver.solve(prompt)`` call.

    ``predicted_answer=None`` and ``confidence='none'`` mean the solver
    declined — the pipeline skips this row entirely and moves on to the
    next generator (teacher LLM, OOD synth, etc.) if configured.

    ``trace_dict`` is an opaque bag passed straight to the matching
    ``CoTRenderer.render``. Solver/renderer pairs agree on its keys;
    the pipeline never inspects it.
    """
    predicted_answer: str | None
    trace_dict: dict[str, Any] = field(default_factory=dict)
    confidence: Confidence = "none"
    method: str = ""


@dataclass
class GeneratedRow:
    """Wire format between stages (JSONL rows under
    ``data/generated/<stage>/rows.jsonl``) and ultimately what
    ``data_merge`` appends to ``data/sources.yaml`` as a consumable
    training source.

    ``source`` tracks which generator produced the row so downstream
    analyses (e.g. "how much of SFT loss came from solver vs teacher?")
    can filter without re-deriving.
    """
    id: str
    prompt: str
    answer: str
    category: str
    cot: str
    source: str                     # "solver" | "teacher_llm" | "ood_augment" | ...
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratedRow":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            answer=d["answer"],
            category=d["category"],
            cot=d["cot"],
            source=d["source"],
            metadata=d.get("metadata", {}),
        )


# ── Protocols ────────────────────────────────────────────────────────────

@runtime_checkable
class Solver(Protocol):
    """Deterministic solver for one category.

    Implementations are typically pure-Python brute-force search,
    constraint propagation, or closed-form inversion. No GPU, no LLM.
    """
    category: str

    def solve(self, prompt: str) -> SolverResult: ...


@runtime_checkable
class Verifier(Protocol):
    """Equality check between predicted and ground-truth answers,
    with category-specific tolerance (numeric ±rel, binary exact, ...).
    """
    category: str

    def check(self, pred: str, gt: str) -> bool: ...


@runtime_checkable
class CoTRenderer(Protocol):
    """Render a ``SolverResult.trace_dict`` into an SFT-ready CoT string.

    The pipeline post-processes the returned string (forces
    ``\\boxed{answer}`` + optionally injects ``[verify]: PASS``) — the
    renderer itself is free to be template-based, LLM-rewritten, or
    anything in between.
    """
    category: str

    def render(self, prompt: str, answer: str, trace: dict[str, Any]) -> str: ...


@runtime_checkable
class DataSynthGenerator(Protocol):
    """OOD / synthetic puzzle generator for one category (Phase 3).

    Emits ``GeneratedRow`` objects directly — no solver pass needed,
    because the rule is chosen upfront so the answer is known.
    """
    category: str

    def generate(
        self,
        recipe: dict[str, Any],
        n: int,
        *,
        seed: int = 0,
    ) -> Iterable[GeneratedRow]: ...


__all__ = [
    "Confidence",
    "CoTRenderer",
    "DataSynthGenerator",
    "GeneratedRow",
    "Solver",
    "SolverResult",
    "TrainingExample",
    "Verifier",
]
