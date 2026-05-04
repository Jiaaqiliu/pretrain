"""Quality Plan checkpoints — backend-viewer contract.

Checkpoints are **critical-path gates** in a training run's lifecycle
(Plan → Data → Model → ... → Submit). The backend uses them to decide
whether to start or halt a cycle; the viewer renders them in the
cockpit ledger with Sign buttons (manual mode) or auto-status (auto
mode).

Two layers of ownership:

  1. **Declaration** lives in the workspace as
     ``<workspace>/checkpoints.yaml``. Swap the file to get a different
     lifecycle for a different benchmark; missing file ⇒ no gates.
     See :func:`load_slot_decls`.

  2. **State** is a fold over ``memory/records.jsonl``. The fold reads
     two kinds of record:

     * ``checkpoint_event`` — ``event:signoff`` / ``event:reopen`` /
       ``event:evidence_attached``. Written by orchestrator tools or the
       viewer's Sign endpoint; moves the slot to a terminal state.
     * ``checkpoint_review`` — verdicts from the **reviewer** role. This
       is the QA channel: the reviewer cites evidence records, judges
       their quality, and declares whether the slot is
       ``evidence_attached``, ``ready_to_sign``, ``insufficient``, or
       ``reject``. The fold advances the slot's state accordingly.

Two modes, selected by ``NEMO_MAS_CHECKPOINT_MODE``:

  * ``manual`` (default) — orchestrator cannot self-sign; when a
    reviewer declares ``ready_to_sign`` the slot goes to
    ``pending_human`` and the cycle halts. A human clicks Sign in the
    viewer, which appends a ``checkpoint_event`` with
    ``event:signoff, actor:human:<name>``.
  * ``auto`` — the orchestrator / reviewer can sign via ``checkpoint_sign``
    once evidence is in place.

Slot states:

  * ``pending``            — no review yet, not actionable.
  * ``pending_evidence``   — reviewer saw evidence but not enough to sign.
  * ``pending_human``      — manual mode, reviewer said ``ready_to_sign``
                              but no human has yet clicked Sign.
  * ``signed``             — terminal.
  * ``reopened``           — was signed, then reopened (e.g. via
                              ``event:reopen`` or reviewer verdict
                              ``reject``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)


CHECKPOINT_MODE_MANUAL = "manual"
CHECKPOINT_MODE_AUTO = "auto"
VALID_MODES = frozenset({CHECKPOINT_MODE_MANUAL, CHECKPOINT_MODE_AUTO})

TERMINAL_STATES = frozenset({"signed", "reopened"})

# Reviewer verdicts recognised by the fold. Anything else is treated as
# ``pending_evidence`` (best-effort forward-compat for future verdicts).
VERDICT_EVIDENCE_ATTACHED = "evidence_attached"
VERDICT_READY_TO_SIGN = "ready_to_sign"
VERDICT_INSUFFICIENT = "insufficient"
VERDICT_REJECT = "reject"
VALID_VERDICTS = frozenset({
    VERDICT_EVIDENCE_ATTACHED,
    VERDICT_READY_TO_SIGN,
    VERDICT_INSUFFICIENT,
    VERDICT_REJECT,
})


# ── Slot declarations ───────────────────────────────────────────────

def load_slot_decls(workspace_root: Path | str | None) -> list[dict[str, Any]]:
    """Load the ``checkpoints.yaml`` declaration list from a workspace.

    Returns ``[]`` if the workspace has no file — this is the
    "no-checkpoints" mode (useful for benchmarks that don't want a
    Quality Plan). Caller should pass ``[]`` to :func:`fold_checkpoints`
    in that case and the fold will return an empty list.
    """
    if workspace_root is None:
        return []
    path = Path(workspace_root) / "checkpoints.yaml"
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "[checkpoints] could not load %s: %s — running with no gates",
            path, exc,
        )
        return []
    slots = data.get("checkpoints") or []
    if not isinstance(slots, list):
        logger.warning(
            "[checkpoints] %s must contain a ``checkpoints:`` list; "
            "got %s — running with no gates",
            path, type(slots).__name__,
        )
        return []
    # Normalise field presence so downstream code can assume keys exist.
    out = []
    for s in slots:
        if not isinstance(s, dict) or "id" not in s:
            logger.warning("[checkpoints] skipping malformed slot: %r", s)
            continue
        out.append({
            "id": s["id"],
            "short": s.get("short", s["id"]),
            "title": s.get("title", s["id"]),
            "type": s.get("type", "generic"),
            "requires_evidence": list(s.get("requires_evidence") or []),
            "depends_on": list(s.get("depends_on") or []),
            "required": bool(s.get("required", True)),
            "signers": s.get("signers", "owner"),
        })
    return out


# ── Fold ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FoldedSlot:
    id: str
    short: str
    title: str
    type: str
    template: str
    signers: str
    required: bool
    depends_on: tuple[str, ...]
    requires_evidence: tuple[str, ...]
    state: str                              # pending | pending_evidence | pending_human | signed | reopened
    evidence_counts: dict[str, int]         # kind -> count of records tagged checkpoint:<this>
    last_event_ts: str                      # ISO or "" if no events
    last_event_actor: str                   # actor tag of last event, or ""
    last_review_verdict: str                # "" if no reviews yet
    last_review_reason: str                 # single-line reason from the latest review
    last_review_cycle: str                  # cycle_id of the latest review
    can_sign: bool                          # True only in manual mode when state == pending_human


def _record_has_tag(rec: Any, tag: str) -> bool:
    tags = getattr(rec, "tags", None)
    if tags is None and isinstance(rec, dict):
        tags = rec.get("tags") or ()
    return tag in (tags or ())


def _record_get(rec: Any, field: str, default: Any = "") -> Any:
    if isinstance(rec, dict):
        return rec.get(field, default)
    return getattr(rec, field, default)


def _tag_value(rec: Any, prefix: str) -> str:
    tags = _record_get(rec, "tags", ()) or ()
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix):]
    return ""


def _reason_excerpt(body: Any, limit: int = 240) -> str:
    """First line of the review body, trimmed. Guards against non-str bodies."""
    if not isinstance(body, str) or not body:
        return ""
    first = body.strip().splitlines()[0] if body.strip() else ""
    if len(first) > limit:
        return first[:limit] + "…"
    return first


def fold_checkpoints(
    records: Iterable[Any],
    mode: str,
    slots: list[dict[str, Any]] | None = None,
) -> list[FoldedSlot]:
    """Fold ``checkpoint_event`` + ``checkpoint_review`` into per-slot state.

    ``records`` is an iterable of ``MemoryRecord`` dataclasses or plain
    dicts (JSON-decoded from ``records.jsonl``). The fold uses only
    ``kind``, ``tags``, ``ts``, ``body``, ``cycle_id`` fields.

    ``slots`` is the declaration list (from :func:`load_slot_decls`). Pass
    ``None`` to treat the run as having no gates — returns an empty list.

    ``mode`` is compared against :data:`VALID_MODES`; anything unknown
    falls back to manual semantics (fail safe).
    """
    if slots is None or not slots:
        return []
    if mode not in VALID_MODES:
        mode = CHECKPOINT_MODE_MANUAL

    record_list = list(records)
    slot_ids = {s["id"] for s in slots}

    # Bucket checkpoint_event / checkpoint_review by slot id; count
    # evidence records only when tagged with the matching slot id.
    events_by_slot: dict[str, list[Any]] = {sid: [] for sid in slot_ids}
    reviews_by_slot: dict[str, list[Any]] = {sid: [] for sid in slot_ids}
    evidence_by_slot: dict[str, dict[str, int]] = {sid: {} for sid in slot_ids}

    for rec in record_list:
        kind = _record_get(rec, "kind")
        slot_id = _tag_value(rec, "checkpoint:")
        if kind == "checkpoint_event":
            if slot_id in events_by_slot:
                events_by_slot[slot_id].append(rec)
        elif kind == "checkpoint_review":
            if slot_id in reviews_by_slot:
                reviews_by_slot[slot_id].append(rec)
        elif kind and slot_id in evidence_by_slot:
            # Slot-tagged evidence record — count it against this slot.
            bucket = evidence_by_slot[slot_id]
            bucket[kind] = bucket.get(kind, 0) + 1

    # Pass 1: state from checkpoint_event + last review verdict.
    state_by_slot: dict[str, str] = {}
    last_event_ts: dict[str, str] = {}
    last_event_actor: dict[str, str] = {}
    last_review_verdict: dict[str, str] = {}
    last_review_reason: dict[str, str] = {}
    last_review_cycle: dict[str, str] = {}

    for slot in slots:
        sid = slot["id"]
        events = sorted(events_by_slot[sid], key=lambda r: _record_get(r, "ts", ""))
        reviews = sorted(reviews_by_slot[sid], key=lambda r: _record_get(r, "ts", ""))

        state = "pending"
        for e in events:
            if _record_has_tag(e, "event:signoff"):
                state = "signed"
            elif _record_has_tag(e, "event:reopen"):
                state = "reopened"
            elif _record_has_tag(e, "event:evidence_attached") and state == "pending":
                state = "pending_evidence"

        # Reviewer verdicts can lift state further, but never override a
        # terminal ``signed`` / ``reopened`` set by an explicit event.
        if reviews and state not in TERMINAL_STATES:
            latest = reviews[-1]
            verdict = _tag_value(latest, "verdict:")
            last_review_verdict[sid] = verdict
            last_review_reason[sid] = _reason_excerpt(_record_get(latest, "body"))
            last_review_cycle[sid] = _record_get(latest, "cycle_id", "")
            if verdict == VERDICT_EVIDENCE_ATTACHED and state == "pending":
                state = "pending_evidence"
            elif verdict == VERDICT_READY_TO_SIGN:
                state = "pending_evidence"   # further promoted below after deps check
            elif verdict == VERDICT_REJECT:
                # Reviewer explicitly rejected: treat like a reopen.
                state = "reopened"
            # VERDICT_INSUFFICIENT leaves state alone.
        else:
            last_review_verdict[sid] = ""
            last_review_reason[sid] = ""
            last_review_cycle[sid] = ""

        state_by_slot[sid] = state

        if events:
            last = events[-1]
            last_event_ts[sid] = _record_get(last, "ts", "")
            last_event_actor[sid] = _tag_value(last, "actor:")
        else:
            last_event_ts[sid] = ""
            last_event_actor[sid] = ""

    # Pass 2: apply dependency rules + manual-mode ``pending_human`` lift.
    folded: list[FoldedSlot] = []
    for slot in slots:
        sid = slot["id"]
        state = state_by_slot[sid]
        deps_met = all(
            state_by_slot.get(d, "pending") in TERMINAL_STATES
            for d in slot["depends_on"]
        )

        if (state == "pending_evidence"
                and deps_met
                and last_review_verdict.get(sid) == VERDICT_READY_TO_SIGN):
            if mode == CHECKPOINT_MODE_MANUAL and slot["required"]:
                state = "pending_human"
            # In auto mode the signer (reviewer or orchestrator) calls
            # ``checkpoint_sign`` to append a signoff event; the fold
            # picks that up on the next pass. We don't auto-advance
            # state here because signing is an audit-logged act, not a
            # derivation.

        can_sign = (mode == CHECKPOINT_MODE_MANUAL and state == "pending_human")

        folded.append(FoldedSlot(
            id=sid,
            short=slot["short"],
            title=slot["title"],
            type=slot["type"],
            template=f"{slot['type']}_card_v1",
            signers=slot["signers"],
            required=slot["required"],
            depends_on=tuple(slot["depends_on"]),
            requires_evidence=tuple(slot["requires_evidence"]),
            state=state,
            evidence_counts=dict(evidence_by_slot[sid]),
            last_event_ts=last_event_ts[sid],
            last_event_actor=last_event_actor[sid],
            last_review_verdict=last_review_verdict.get(sid, ""),
            last_review_reason=last_review_reason.get(sid, ""),
            last_review_cycle=last_review_cycle.get(sid, ""),
            can_sign=can_sign,
        ))

    return folded


def first_required_blocker(folded: Iterable[FoldedSlot]) -> FoldedSlot | None:
    """Return the first required slot not in a terminal state, or ``None``.

    Order follows the declaration order in ``checkpoints.yaml`` (which is
    also the dependency order by convention). Optional slots never block.
    """
    for slot in folded:
        if slot.required and slot.state not in TERMINAL_STATES:
            return slot
    return None


def evidence_refs_for_slot(
    slot: FoldedSlot | dict,
    records: Iterable[Any],
) -> list[str]:
    """Pick one most-recent slot-tagged ref per required evidence kind.

    Used by the viewer's Sign endpoint (which needs ≥1 ref to satisfy
    ``REF_RULES["checkpoint_event"]``). Only considers records tagged
    ``checkpoint:<this slot's id>`` — mirrors the fold.
    """
    requires = (slot.requires_evidence
                if isinstance(slot, FoldedSlot)
                else tuple(slot.get("requires_evidence") or ()))
    sid = slot.id if isinstance(slot, FoldedSlot) else slot.get("id", "")
    by_kind: dict[str, Any] = {}
    for rec in records:
        kind = _record_get(rec, "kind")
        if kind not in requires:
            continue
        if _tag_value(rec, "checkpoint:") != sid:
            continue
        prev = by_kind.get(kind)
        prev_ts = _record_get(prev, "ts", "") if prev is not None else ""
        cur_ts = _record_get(rec, "ts", "")
        if prev is None or cur_ts >= prev_ts:
            by_kind[kind] = rec
    return [_record_get(by_kind[k], "id", "") for k in requires if k in by_kind]
