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
_CROSS_CUTTING = frozenset({"breakthrough", "failed_attempt"})

KIND_WHITELIST: dict[str, frozenset[str]] = {
    "analyst": frozenset({
        "data_audit_finding",
        "benchmark_rule",
        "profile_run",
        "eval_report",
        "error_pattern",
        "data_gap",
    }) | _CROSS_CUTTING,
    "data_engineer": frozenset({
        "distill_batch",
        "dataset_snapshot",
    }) | _CROSS_CUTTING,
    "theorist": frozenset({
        "hypothesis",
        "recipe_proposal",
    }) | _CROSS_CUTTING,
    "engineer": frozenset({
        "runner_capability",
        "training_run",
        "cv_result",
    }) | _CROSS_CUTTING,
}

# All known kinds (for validation of refs targets, mem_search filters,
# and friendly error messages). Internal kinds prefixed with `_` are
# system records (e.g. links) and never returned by normal queries.
ALL_KINDS: frozenset[str] = (
    KIND_WHITELIST["analyst"]
    | KIND_WHITELIST["data_engineer"]
    | KIND_WHITELIST["theorist"]
    | KIND_WHITELIST["engineer"]
)

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
    "breakthrough":     _require_min_refs(1),
    "recipe_proposal":  _require_ref_kind("eval_report", "data_gap"),
    "training_run":     _require_refs_with_kinds("recipe_proposal", "dataset_snapshot"),
    "cv_result":        _require_ref_kind("training_run"),
    "eval_report":      _require_ref_kind("training_run"),
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
