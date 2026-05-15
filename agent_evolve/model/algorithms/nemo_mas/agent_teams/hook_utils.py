"""Shared helpers for Claude Code hooks, the MCP server, and the trace viewer.

All three need cheap access to the active run's paths.

Env vars consumed (set by the MCP server's ``start_iteration`` tool, or by
the operator before launching ``claude``):

  * ``NEMO_MAS_WORK_DIR``       — run root, e.g. ``runs/nemo-mas-teams-v1``.
  * ``NEMO_MAS_WORKSPACE_ROOT`` — active forked workspace for the current
                                   iteration. Falls back to the seed
                                   workspace if unset.
  * ``NEMO_MAS_MEMORY_PATH``    — path to the cross-cycle ledger
                                   (``<work_dir>/memory/records.jsonl``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def current_work_dir() -> Path | None:
    raw = os.environ.get("NEMO_MAS_WORK_DIR")
    return Path(raw) if raw else None


def current_workspace_root() -> Path | None:
    raw = os.environ.get("NEMO_MAS_WORKSPACE_ROOT")
    return Path(raw) if raw else None


def current_memory_path() -> Path | None:
    raw = os.environ.get("NEMO_MAS_MEMORY_PATH")
    if raw:
        return Path(raw)
    wd = current_work_dir()
    return (wd / "memory" / "records.jsonl") if wd else None


def read_records_jsonl(path: Path | None) -> list[dict[str, Any]]:
    """Parse ``records.jsonl`` as plain dicts.

    Hooks run as short-lived shell-invoked Python scripts; they don't
    need the in-memory ``RecipeMemory`` index, just the raw records for
    one fold. Returns ``[]`` if the file is missing or empty.
    """
    if path is None or not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out




def count_records_of_kind(kind: str) -> int:
    """Count records of ``kind`` in the active ledger.

    Used by the kaggle-budget hook (``kind='kaggle_submission_result'``)
    and optionally by other guards that enforce per-run caps.
    """
    records = read_records_jsonl(current_memory_path())
    return sum(1 for r in records if r.get("kind") == kind)


def meta_path() -> Path | None:
    wd = current_work_dir()
    return (wd / "meta.json") if wd else None


def read_meta() -> dict[str, Any]:
    p = meta_path()
    if p is None or not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def cycle_workspace_path(work_dir: Path | str, cycle_id: str) -> Path:
    """Forked workspace path for ``cycle_id`` under a run's ``work_dir``.

    Single source of truth for the convention shared by ``start_iteration``
    (the producer) and the trace viewer (a read-only consumer). Returns
    ``<work_dir>/cycles/<cycle_id>/.fork_target/nodes/workspace/workspace``.
    The path may not exist yet (pre-fork) — the caller decides how to handle.
    """
    return (
        Path(work_dir) / "cycles" / cycle_id
        / ".fork_target" / "nodes" / "workspace" / "workspace"
    )
