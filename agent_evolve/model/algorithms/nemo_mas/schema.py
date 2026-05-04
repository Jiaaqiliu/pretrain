"""Typed-record schema + per-role write whitelist + ref constraints.

The orchestrator-worker pattern enforces "each role writes different
things" at the schema level — out-of-whitelist ``mem_write`` calls and
ref-constraint violations are rejected before any record is appended.

Constants in this file are the source of truth referenced by ``memory.py``,
``tools.py``, and the workspace's ``tools/<role>.yaml`` ``mem_write_kinds``
sections (which exist for LLM-readable documentation; this file does the
enforcement).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable


# ── Per-role write whitelist ─────────────────────────────────────────

# "any" pseudo-role — kinds any worker can write (with constraints).
# checkpoint_event is here so every role may attach evidence or reopen a slot;
# signoff by an agent is additionally gated by the checkpoint_sign tool.
_CROSS_CUTTING = frozenset({"breakthrough", "failed_attempt", "checkpoint_event"})

# Orchestrator may write directive_response and checkpoint_event (the latter
# only through its `checkpoint_sign` tool in auto mode, which enforces the
# evidence precondition). It cannot write any worker-kind record.
_ORCHESTRATOR_AUTO_KINDS = frozenset({"directive_response", "checkpoint_event"})

KIND_WHITELIST: dict[str, frozenset[str]] = {
    "reviewer": frozenset({
        "data_audit_finding",
        "benchmark_rule",
        "profile_run",
        "eval_report",
        "error_pattern",
        "data_gap",
        "directive_response",
        # QA officer duty: verdicts on Quality Plan checkpoint evidence.
        # Exactly one kind per role owns this; the reviewer is it.
        "checkpoint_review",
        # Result of a kaggle_submit call — the reviewer pipes the CLI's
        # submission_id + initial status into memory.
        "kaggle_submission_result",
    }) | _CROSS_CUTTING,
    "data_worker": frozenset({
        "distill_batch",
        "dataset_snapshot",
        "directive_response",
    }) | _CROSS_CUTTING,
    "planner": frozenset({
        "hypothesis",
        "recipe_proposal",
        "directive_response",
    }) | _CROSS_CUTTING,
    "trainer": frozenset({
        "training_run",
        "cv_result",
        "directive_response",
        # Packaged LoRA adapter zip ready for Kaggle submission. Produced
        # by pack_submission; body contains zip path + adapter_config
        # summary. Reviewer cites this when posting cp_submission_ready.
        "submission_artifact",
    }) | _CROSS_CUTTING,
    # Pseudo-role used when the orchestrator's checkpoint_sign /
    # directive_respond tools write records on its behalf. Not a spawnable
    # worker role.
    "orchestrator_auto": _ORCHESTRATOR_AUTO_KINDS,
}

# All known kinds (for validation of refs targets, mem_search filters,
# and friendly error messages). ``human_directive`` is not in any role's
# whitelist because only the viewer writes it; it is still a valid ref
# target, so it belongs here. Internal kinds prefixed with `_` are
# system records (e.g. links) and never returned by normal queries.
ALL_KINDS: frozenset[str] = (
    KIND_WHITELIST["reviewer"]
    | KIND_WHITELIST["data_worker"]
    | KIND_WHITELIST["planner"]
    | KIND_WHITELIST["trainer"]
    | KIND_WHITELIST["orchestrator_auto"]
    | frozenset({"human_directive"})
)

# Kinds writable only by the frontend (not through any agent role).
# ``validate_record`` honors these when ``author`` starts with ``human:``.
HUMAN_KINDS: frozenset[str] = frozenset({"human_directive"})

INTERNAL_KINDS: frozenset[str] = frozenset({"_link"})


# ── Ref constraints ──────────────────────────────────────────────────

# Each rule takes (record, kind_lookup) where kind_lookup(rec_id) -> str|None
# (the kind of the referenced record, or None if not found). Rule returns
# None on success or an error string on failure.

RefLookup = Callable[[str], str | None]
RefRule = Callable[["MemoryRecord", RefLookup], str | None]


def _require_min_refs(n: int) -> RefRule:
    def rule(rec: "MemoryRecord", _lookup: RefLookup) -> str | None:
        if len(rec.refs) < n:
            return f"kind={rec.kind!r} requires at least {n} ref(s); got {len(rec.refs)}."
        return None
    return rule


def _require_ref_kind(*allowed: str) -> RefRule:
    allowed_set = frozenset(allowed)

    def rule(rec: "MemoryRecord", lookup: RefLookup) -> str | None:
        if not rec.refs:
            return f"kind={rec.kind!r} requires at least one ref to {sorted(allowed_set)}; got none."
        kinds = [lookup(r) for r in rec.refs]
        if not any(k in allowed_set for k in kinds):
            present = [k for k in kinds if k]
            return (
                f"kind={rec.kind!r} requires at least one ref to one of "
                f"{sorted(allowed_set)}; refs resolve to {present}."
            )
        return None
    return rule


def _require_refs_with_kinds(*required: str) -> RefRule:
    """Refs must collectively cover EVERY listed kind (one each, not OR)."""
    def rule(rec: "MemoryRecord", lookup: RefLookup) -> str | None:
        kinds = {lookup(r) for r in rec.refs}
        missing = [k for k in required if k not in kinds]
        if missing:
            return (
                f"kind={rec.kind!r} requires refs to all of {list(required)}; "
                f"missing {missing}."
            )
        return None
    return rule


def _chain(*rules: RefRule) -> RefRule:
    def rule(rec: "MemoryRecord", lookup: RefLookup) -> str | None:
        for r in rules:
            err = r(rec, lookup)
            if err:
                return err
        return None
    return rule


REF_RULES: dict[str, RefRule] = {
    "breakthrough":        _require_min_refs(1),
    "recipe_proposal":     _require_ref_kind("eval_report", "data_gap"),
    "training_run":        _require_refs_with_kinds("recipe_proposal", "dataset_snapshot"),
    "cv_result":           _require_ref_kind("training_run"),
    "eval_report":         _require_ref_kind("training_run"),
    # A signoff/reopen/evidence event must point at something — usually the
    # evidence records that justify it. Minimum one ref; the slot-specific
    # "covers all requires_evidence kinds" check is done by the signing tool,
    # not here (this is schema, not workflow).
    "checkpoint_event":    _require_min_refs(1),
    # QA verdict from the reviewer role. MUST cite the evidence records it
    # is judging — no refs means "no evidence reviewed", which is useless.
    "checkpoint_review":   _require_min_refs(1),
    # Every orchestrator reply must point at the human_directive it answers.
    "directive_response":  _require_ref_kind("human_directive"),
    # Packaged adapter zip must trace back to a training_run so reviewers
    # can audit which checkpoint shipped.
    "submission_artifact": _require_ref_kind("training_run"),
    # Kaggle submission result must trace back to the artifact we pushed.
    "kaggle_submission_result": _require_ref_kind("submission_artifact"),
    # All other kinds: no ref requirements (refs are optional but recommended).
}


# ── Record dataclass ────────────────────────────────────────────────

_ID_RE = re.compile(r"^rec_[0-9a-f]{6,}$")


@dataclass(frozen=True)
class MemoryRecord:
    id: str                              # rec_<hex>
    cycle_id: str                        # "0007" or "init"
    author: str                          # role name; auto-filled by mem_write
    kind: str                            # see ALL_KINDS / INTERNAL_KINDS
    title: str
    body: str
    tags: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()
    ts: str = ""                         # ISO-8601, set by memory store

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cycle_id": self.cycle_id,
            "author": self.author,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "refs": list(self.refs),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(
            id=d["id"],
            cycle_id=d.get("cycle_id", ""),
            author=d.get("author", ""),
            kind=d["kind"],
            title=d.get("title", ""),
            body=d.get("body", ""),
            tags=tuple(d.get("tags") or ()),
            refs=tuple(d.get("refs") or ()),
            ts=d.get("ts", ""),
        )


# ── Validation ──────────────────────────────────────────────────────

class RecordValidationError(ValueError):
    """Raised when a candidate record fails kind whitelist or ref rules."""


def validate_record(
    record: MemoryRecord,
    *,
    role: str,
    ref_lookup: RefLookup,
) -> None:
    """Raise RecordValidationError if the record is invalid for this role.

    Checks (in order):
      1. ID format.
      2. Title and body non-empty.
      3. Kind is in this role's whitelist (or an internal kind).
      4. Refs are well-formed ids that resolve (each ref id resolves via
         ref_lookup; unresolvable refs are an error).
      5. Per-kind ref rules from REF_RULES.
    """
    if not _ID_RE.match(record.id):
        raise RecordValidationError(
            f"record id {record.id!r} must match pattern rec_<hex>"
        )
    if not record.title.strip():
        raise RecordValidationError("record.title must be non-empty")
    if not record.body.strip():
        raise RecordValidationError("record.body must be non-empty")

    if record.kind in INTERNAL_KINDS:
        return  # internal records bypass role whitelist + ref rules

    # Frontend-authored records: viewer's /directive endpoint writes these
    # with ``author="human:<name>"``. Skip role-whitelist (there is no role),
    # still enforce kind membership and per-kind ref rules below.
    if record.author.startswith("human:"):
        if record.kind not in HUMAN_KINDS:
            raise RecordValidationError(
                f"author={record.author!r} may only write "
                f"{sorted(HUMAN_KINDS)}; got kind={record.kind!r}"
            )
        rule = REF_RULES.get(record.kind)
        if rule is not None:
            err = rule(record, ref_lookup)
            if err:
                raise RecordValidationError(err)
        return

    role_kinds = KIND_WHITELIST.get(role)
    if role_kinds is None:
        raise RecordValidationError(
            f"unknown role {role!r}; expected one of {sorted(KIND_WHITELIST)}"
        )
    if record.kind not in role_kinds:
        raise RecordValidationError(
            f"role={role!r} not allowed to write kind={record.kind!r}; "
            f"allowed: {sorted(role_kinds)}"
        )

    for ref_id in record.refs:
        if ref_lookup(ref_id) is None:
            raise RecordValidationError(
                f"ref {ref_id!r} does not resolve to an existing record"
            )

    rule = REF_RULES.get(record.kind)
    if rule is not None:
        err = rule(record, ref_lookup)
        if err:
            raise RecordValidationError(err)


def kinds_for_role(role: str) -> Iterable[str]:
    """Helper for building tool descriptions that list allowed kinds."""
    return sorted(KIND_WHITELIST.get(role, frozenset()))
