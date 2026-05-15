"""nemo_mas — Multi-agent system for the Nvidia Nemotron Reasoning Kaggle.

The interactive Claude Code Agent Teams runtime is the only entry point.
Workers (planner, data_worker, trainer) coordinate through a typed-record
shared ledger with BM25 search, executing through the Bash CLI and the
MCP server in ``agent_teams/``.

Public API exposed here is the data layer only — memory, schema, and the
record validation rules. The runtime itself lives in ``agent_teams/``.
"""

from __future__ import annotations

from .memory import RecipeMemory
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
    "RecipeMemory",
    "RecordValidationError",
    "REF_RULES",
    "validate_record",
]
