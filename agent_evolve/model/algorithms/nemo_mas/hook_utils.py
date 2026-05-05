"""Shared helpers for Claude Code hooks and the MCP server.

Hooks and the MCP server both need to answer questions like
"is any required checkpoint currently blocking progress?" and
"how many kaggle submits have already been made this run?" — without
spinning up the whole ``NemoMASAlgorithm`` just to fold the ledger.

Env vars consumed (set by the MCP server's ``start_iteration`` tool, or by
the operator before launching ``claude``):

  * ``NEMO_MAS_WORK_DIR``       — run root, e.g. ``runs/nemo-mas-teams-v1``.
  * ``NEMO_MAS_WORKSPACE_ROOT`` — active forked workspace for the current
                                   iteration. Falls back to the seed
                                   workspace if unset.
  * ``NEMO_MAS_MEMORY_PATH``    — path to the cross-cycle ledger
                                   (``<work_dir>/memory/records.jsonl``).
  * ``NEMO_MAS_CHECKPOINT_MODE`` — ``manual`` (default) or ``auto``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .checkpoints import (
    CHECKPOINT_MODE_AUTO,
    CHECKPOINT_MODE_MANUAL,
    FoldedSlot,
    first_required_blocker,
    fold_checkpoints,
    load_slot_decls,
)


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


def current_checkpoint_mode() -> str:
    mode = os.environ.get("NEMO_MAS_CHECKPOINT_MODE", CHECKPOINT_MODE_MANUAL)
    return mode if mode in (CHECKPOINT_MODE_AUTO, CHECKPOINT_MODE_MANUAL) else CHECKPOINT_MODE_MANUAL


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


def fold_current_run() -> tuple[list[FoldedSlot], str]:
    """Fold the active ledger against the active workspace's slots.

    Returns ``(folded, mode)``. Empty list when no workspace is resolvable
    or the workspace has no ``checkpoints.yaml``.
    """
    ws = current_workspace_root()
    records = read_records_jsonl(current_memory_path())
    mode = current_checkpoint_mode()
    slots = load_slot_decls(ws)
    if not slots:
        return ([], mode)
    folded = fold_checkpoints(records, mode, slots=slots)
    return (folded, mode)


def first_blocker_or_none() -> FoldedSlot | None:
    folded, _mode = fold_current_run()
    return first_required_blocker(folded) if folded else None


def format_blocker_message(blocker: FoldedSlot, mode: str) -> str:
    """Human-readable blocker text for printing to a hook's stderr.

    Mirrors the shape of ``orchestrator._format_blocker_block`` but
    trimmed for a 2KB hook message budget.
    """
    ev_counts = ", ".join(
        f"{k}={v}" for k, v in sorted(blocker.evidence_counts.items())
    ) or "(none)"
    lines = [
        f"[nemo_mas] BLOCKED on required checkpoint {blocker.id} ({blocker.title}).",
        f"  state: {blocker.state}",
        f"  requires_evidence: {list(blocker.requires_evidence) or '(none)'}",
        f"  evidence_counts: {ev_counts}",
        f"  depends_on: {list(blocker.depends_on) or '(none)'}",
    ]
    if blocker.last_review_verdict:
        lines.append(
            f"  last_review: verdict={blocker.last_review_verdict} · "
            f"cycle {blocker.last_review_cycle} · {blocker.last_review_reason}"
        )
    if blocker.state == "pending_human" and mode == CHECKPOINT_MODE_MANUAL:
        lines.append(
            f"  ACTION: tell the lead `sign {blocker.id} with refs=[...]` "
            "once you've reviewed the attached evidence. Required slots "
            "cannot be skipped."
        )
    elif blocker.state == "pending_evidence":
        lines.append(
            "  ACTION: produce evidence records tagged "
            f"`checkpoint:{blocker.id}` covering the requires_evidence "
            "kinds; have the reviewer post a verdict."
        )
    else:
        lines.append(
            f"  ACTION: progress slot {blocker.id!r} to a terminal state "
            "before creating further tasks."
        )
    return "\n".join(lines)


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
