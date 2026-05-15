"""RecipeMemory — append-only typed-record store with BM25 search.

Storage:
  - ``records.jsonl`` under ``memory/``: one JSON line per record with
    all scalar metadata (id, cycle, author, kind, title, tags, refs, ts)
    and a ``body_path`` pointer. Bodies are NOT inlined.
  - ``records/<role>/<kind>/<id>.md`` under ``memory/``: the full record
    body as a standalone markdown file. Grep-friendly; browsable.

Write path: (1) write body .md, (2) append JSONL line. Crash between
those leaves an orphan .md (recoverable by rebuilding JSONL from files)
rather than a dangling pointer.

Read path: on ``_load``, each line is parsed; if ``body_path`` is set,
the body is hydrated from disk. Legacy records with inline ``body`` and
no ``body_path`` still work unchanged.

Internal ``_link`` records are interleaved (added by post-hoc linking
via ``link()``) and filtered from normal queries.

BM25: vendored Okapi BM25 (no external dep). Index is rebuilt on load and
on each write. The corpus is expected to fit in memory (<100k records);
beyond that, switch to an actual BM25 library.

Field weights for BM25 indexing:
  - title × 3
  - tags × 2
  - body × 1

This matches the design in DESIGN.md §3 and the prompt instructions
written in ``prompts/<role>.md``.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schema import (
    INTERNAL_KINDS,
    MemoryRecord,
    RecordValidationError,
    validate_record,
)


# ── BM25 (vendored, ~50 lines) ───────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


@dataclass
class _BM25:
    """Okapi BM25. No external deps. Suitable for <100k docs."""
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.docs: list[list[str]] = []          # parallel to ids
        self.ids: list[str] = []
        self.doc_len: list[int] = []
        self.df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        self.avgdl: float = 0.0

    def fit(self, ids: list[str], docs: list[list[str]]) -> None:
        self.ids = list(ids)
        self.docs = list(docs)
        self.tf = [Counter(d) for d in docs]
        self.doc_len = [len(d) for d in docs]
        self.df = Counter()
        for tokens in (set(d) for d in docs):
            for t in tokens:
                self.df[t] += 1
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

    def add(self, id_: str, doc: list[str]) -> None:
        self.ids.append(id_)
        self.docs.append(doc)
        tf = Counter(doc)
        self.tf.append(tf)
        self.doc_len.append(len(doc))
        for t in set(doc):
            self.df[t] += 1
        n = len(self.doc_len)
        self.avgdl = sum(self.doc_len) / n if n else 0.0

    def score_query(
        self,
        query: list[str],
        candidate_idx: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return [(idx, score)] sorted by score desc, restricted to candidate_idx."""
        n = len(self.docs)
        if n == 0 or not query:
            return []
        idxs = candidate_idx if candidate_idx is not None else list(range(n))
        scores: list[tuple[int, float]] = []
        for i in idxs:
            if i < 0 or i >= n:
                continue
            s = 0.0
            tf = self.tf[i]
            dl = self.doc_len[i] or 1
            for q in query:
                f = tf.get(q, 0)
                if f == 0:
                    continue
                df = self.df.get(q, 0)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += idf * (f * (self.k1 + 1.0)) / denom
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def _doc_text(rec: MemoryRecord) -> list[str]:
    """Build the BM25 token stream for one record with field weights baked in
    via repetition (×3 for title, ×2 for tags, ×1 for body).

    Repeating tokens is the simplest way to weight fields under standard
    BM25 without changing the formula.
    """
    title_toks = _tokenize(rec.title)
    tag_toks = [t for tag in rec.tags for t in _tokenize(tag)]
    body_toks = _tokenize(rec.body)
    return (title_toks * 3) + (tag_toks * 2) + body_toks


# ── RecipeMemory ─────────────────────────────────────────────────────


class RecipeMemory:
    """Append-only typed-record store with BM25 search.

    Thread-safe writes via a single lock; concurrent reads are fine.

    Persistence:
      - records.jsonl: one JSON object per line.
      - On load(): parses all lines, builds in-memory list + BM25 index.
      - On write(): validates, generates id + ts, appends to JSONL,
        updates in-memory list + BM25 index incrementally.
    """

    def __init__(self, records_path: Path | str):
        self.records_path = Path(records_path)
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        # Bodies live in ``<records_path>.parent/records/<role>/<kind>/<id>.md``.
        self._body_root = self.records_path.parent / "records"
        self._body_root.mkdir(parents=True, exist_ok=True)
        self._records: list[MemoryRecord] = []
        self._by_id: dict[str, MemoryRecord] = {}
        self._idx_by_id: dict[str, int] = {}
        self._bm25 = _BM25()
        self._lock = threading.Lock()
        self._cycle_id: str = "init"
        self._load()

    # ── body-file helpers ───────────────────────────────────────

    def _body_rel_path(self, role: str, kind: str, rec_id: str) -> str:
        """Workspace-relative body path as stored in the JSONL pointer."""
        return f"records/{role}/{kind}/{rec_id}.md"

    def _body_abs_path(self, rel: str) -> Path:
        """Resolve a stored body_path against this ledger's body root."""
        # Defensive: ignore any leading "records/" since _body_root already
        # has that component. Accept either shape on read for forward compat.
        if rel.startswith("records/"):
            return self._body_root / rel[len("records/"):]
        return self._body_root / rel

    def _read_body(self, rel: str) -> str:
        try:
            return self._body_abs_path(rel).read_text(encoding="utf-8")
        except OSError:
            return ""  # body file missing — leave empty rather than crashing

    def _write_body(self, rel: str, body: str) -> None:
        p = self._body_abs_path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")

    # ── lifecycle ────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.records_path.exists():
            self._bm25.fit([], [])
            return
        ids: list[str] = []
        docs: list[list[str]] = []
        with self.records_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = MemoryRecord.from_dict(d)
                # Hydrate body from disk when the JSONL line carries only a
                # pointer. Records migrated from pre-split format keep their
                # inline body and skip this branch.
                if rec.body_path and not rec.body:
                    rec = MemoryRecord(
                        id=rec.id, cycle_id=rec.cycle_id, author=rec.author,
                        kind=rec.kind, title=rec.title,
                        body=self._read_body(rec.body_path),
                        tags=rec.tags, refs=rec.refs, ts=rec.ts,
                        body_path=rec.body_path,
                    )
                self._records.append(rec)
                self._by_id[rec.id] = rec
                self._idx_by_id[rec.id] = len(self._records) - 1
                ids.append(rec.id)
                docs.append(_doc_text(rec))
        self._bm25.fit(ids, docs)

    def set_cycle_id(self, cycle_id: str) -> None:
        """Stamp subsequent writes with this cycle_id (pure metadata)."""
        self._cycle_id = str(cycle_id)

    def cycle_id(self) -> str:
        return self._cycle_id

    # ── write ────────────────────────────────────────────────────

    def write(
        self,
        *,
        role: str,
        kind: str,
        title: str,
        body: str,
        tags: Iterable[str] = (),
        refs: Iterable[str] = (),
    ) -> MemoryRecord:
        """Validate + append + index. Returns the persisted MemoryRecord.

        Raises RecordValidationError if the role isn't allowed to write
        this kind, or any per-kind ref rule fails.

        Persistence: the body is written to a standalone markdown file at
        ``records/<role>/<kind>/<id>.md``; the JSONL line carries only
        scalar metadata and a ``body_path`` pointer. The .md file is
        written FIRST so a crash before the JSONL append leaves an orphan
        file (recoverable) rather than a dangling pointer.
        """
        rec_id = _new_id()
        body_rel = self._body_rel_path(role, kind, rec_id)
        rec = MemoryRecord(
            id=rec_id,
            cycle_id=self._cycle_id,
            author=role,
            kind=kind,
            title=title,
            body=body,
            tags=tuple(tags),
            refs=tuple(refs),
            ts=_now_iso(),
            body_path=body_rel,
        )
        validate_record(rec, role=role, ref_lookup=self._ref_lookup)
        with self._lock:
            # Body file first — crash-safe ordering: a dangling JSONL
            # pointer is worse than an orphan .md.
            self._write_body(body_rel, body)
            self._records.append(rec)
            self._by_id[rec.id] = rec
            self._idx_by_id[rec.id] = len(self._records) - 1
            self._bm25.add(rec.id, _doc_text(rec))
            with self.records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(inline_body=False)) + "\n")
        return rec

    def _ref_lookup(self, rec_id: str) -> str | None:
        rec = self._by_id.get(rec_id)
        return rec.kind if rec is not None else None

    # ── read ─────────────────────────────────────────────────────

    def get(self, rec_id: str) -> MemoryRecord | None:
        return self._by_id.get(rec_id)

    def recent(
        self,
        *,
        kind: str | None = None,
        author: str | None = None,
        tags: Iterable[str] | None = None,
        k: int = 10,
    ) -> list[MemoryRecord]:
        tag_set = frozenset(tags) if tags else None
        out: list[MemoryRecord] = []
        for rec in reversed(self._records):
            if rec.kind in INTERNAL_KINDS:
                continue
            if kind and rec.kind != kind:
                continue
            if author and rec.author != author:
                continue
            if tag_set and not tag_set.issubset(set(rec.tags)):
                continue
            out.append(rec)
            if len(out) >= k:
                break
        return out

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        author: str | None = None,
        tags: Iterable[str] | None = None,
        cycle_range: tuple[str, str] | None = None,
        top_k: int = 8,
    ) -> list[tuple[MemoryRecord, float]]:
        tag_set = frozenset(tags) if tags else None
        candidate_idx: list[int] = []
        for i, rec in enumerate(self._records):
            if rec.kind in INTERNAL_KINDS:
                continue
            if kind and rec.kind != kind:
                continue
            if author and rec.author != author:
                continue
            if tag_set and not tag_set.issubset(set(rec.tags)):
                continue
            if cycle_range:
                lo, hi = cycle_range
                if not (lo <= rec.cycle_id <= hi):
                    continue
            candidate_idx.append(i)

        q_tokens = _tokenize(query)
        scored = self._bm25.score_query(q_tokens, candidate_idx=candidate_idx)
        out: list[tuple[MemoryRecord, float]] = []
        for idx, score in scored[:top_k]:
            out.append((self._records[idx], score))
        return out

    def link(self, child_id: str, parent_id: str, *, role: str = "_system",
             relation: str = "refs") -> MemoryRecord:
        """Append a ``_link`` record. Append-only: the original child's
        ``refs`` field is not mutated, but readers can collect _link records
        to compute the augmented ref graph."""
        if child_id not in self._by_id:
            raise RecordValidationError(f"child id {child_id!r} not found")
        if parent_id not in self._by_id:
            raise RecordValidationError(f"parent id {parent_id!r} not found")
        rec_id = _new_id()
        body = f"relation={relation}\nchild={child_id}\nparent={parent_id}"
        body_rel = self._body_rel_path(role, "_link", rec_id)
        rec = MemoryRecord(
            id=rec_id,
            cycle_id=self._cycle_id,
            author=role,
            kind="_link",
            title=f"link {child_id}->{parent_id}",
            body=body,
            tags=(relation,),
            refs=(child_id, parent_id),
            ts=_now_iso(),
            body_path=body_rel,
        )
        # Internal kind → bypass whitelist; still requires resolvable refs
        # (which we just verified).
        with self._lock:
            self._write_body(body_rel, body)
            self._records.append(rec)
            self._by_id[rec.id] = rec
            self._idx_by_id[rec.id] = len(self._records) - 1
            self._bm25.add(rec.id, _doc_text(rec))
            with self.records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(inline_body=False)) + "\n")
        return rec

    def all_records(self) -> list[MemoryRecord]:
        """Snapshot of all records (excluding internal _link kinds)."""
        return [r for r in self._records if r.kind not in INTERNAL_KINDS]

    def breakthroughs_block(self, max_chars: int = 4000) -> str:
        """Format all breakthroughs into a prompt-injectable block.

        The orchestrator and all workers receive this in their system
        prompt so global priors are always in context.
        """
        items = [r for r in self._records if r.kind == "breakthrough"]
        if not items:
            return ""
        lines = ["# Breakthroughs (global priors — applies every cycle)\n"]
        budget = max_chars
        for r in items:
            chunk = (
                f"\n## {r.id} (cycle {r.cycle_id}, by {r.author})\n"
                f"**{r.title}**\n\n{r.body}\n"
            )
            if len(chunk) > budget:
                break
            lines.append(chunk)
            budget -= len(chunk)
        return "".join(lines)


# ── helpers ──────────────────────────────────────────────────────────


def _new_id() -> str:
    return f"rec_{secrets.token_hex(6)}"


def _now_iso() -> str:
    # Compact ISO-8601 (UTC, seconds resolution).
    t = time.gmtime()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", t)
