"""nemo_mas — Orchestrator-worker MAS algorithm for Nemotron Reasoning.

A no-arg-constructible alternative to ``mcgs`` in ``TRAINING_ALGORITHMS``.
The orchestrator and four workers (Analyst, DataEngineer, Theorist,
Engineer) coordinate through a typed-record memory store with BM25
search. The algorithm itself never trains, evaluates, or generates data
— it brokers LLM workers, which call backend tools provided by the
caller.

Entry point: ``NemoMASAlgorithm.run_cycle(ctx) -> MCGSCycleReport``.

See ``seed_workspaces/nemo_mas_reasoner/DESIGN.md`` for the full design.
"""

from __future__ import annotations

from .memory import RecipeMemory
from .orchestrator import NemoMASAlgorithm
from .schema import (
    KIND_WHITELIST,
    REF_RULES,
    MemoryRecord,
    RecordValidationError,
    validate_record,
)

__all__ = [
    "KIND_WHITELIST",
    "MemoryRecord",
    "NemoMASAlgorithm",
    "RecipeMemory",
    "RecordValidationError",
    "REF_RULES",
    "validate_record",
]
