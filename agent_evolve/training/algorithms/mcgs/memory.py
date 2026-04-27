"""NodeMemoryStore — persistent memory of past trials.

v1 uses a keyword (BM25-lite) retriever. FAISS is deferred.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ...types import TrainingSearchNode, WorkspaceMutation


_WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


class NodeMemoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root).mkdir(parents=True, exist_ok=True)
        self.node_records = self.root / "node_records.jsonl"
        self.successful = self.root / "successful_mutations.jsonl"
        self.failed = self.root / "failed_mutations.jsonl"
        self.fusion_candidates = self.root / "fusion_candidates.jsonl"

    # ── Writes ──────────────────────────────────────────────────────

    def record(self, node: TrainingSearchNode) -> None:
        row = _node_to_record(node)
        self._append(self.node_records, row)
        if node.is_valid is True:
            self._append(self.successful, row)
        elif node.is_valid is False:
            self._append(self.failed, row)

    def record_fusion_candidate(self, mutation: WorkspaceMutation) -> None:
        self._append(self.fusion_candidates, asdict(mutation))

    # ── Reads / retrieval ───────────────────────────────────────────

    def all_records(self, path: Path | None = None) -> list[dict]:
        target = path or self.node_records
        if not target.exists():
            return []
        rows: list[dict] = []
        with open(target) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def retrieve(
        self,
        query: str,
        k: int = 5,
        source: Path | None = None,
    ) -> list[dict]:
        """Return the top-k records scored against ``query`` with BM25-lite."""
        rows = self.all_records(source)
        if not rows:
            return []
        query_tokens = _tokens(query)
        doc_tokens = [_tokens(_doc_text(row)) for row in rows]
        scores = _bm25(query_tokens, doc_tokens)
        ranked = sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)
        return [row for _, row in ranked[:k]]

    # ── Private ─────────────────────────────────────────────────────

    def _append(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")


# ── Helpers ──────────────────────────────────────────────────────────────

def _doc_text(row: dict) -> str:
    parts = [
        str(row.get("mutation_type", "")),
        str(row.get("summary", row.get("mutation_plan", ""))),
        " ".join(str(k) for k in (row.get("error_buckets") or {}).keys()),
    ]
    return " ".join(parts)


def _bm25(query: Iterable[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    if not docs:
        return []
    avgdl = sum(len(d) for d in docs) / len(docs) or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for term in set(d):
            df[term] = df.get(term, 0) + 1
    n_docs = len(docs)
    scores = [0.0] * n_docs
    for term in query:
        if term not in df:
            continue
        idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
        for i, d in enumerate(docs):
            freq = d.count(term)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * (len(d) / avgdl))
            scores[i] += idf * (freq * (k1 + 1)) / denom
    return scores


def _node_to_record(node: TrainingSearchNode) -> dict:
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "mutation_type": getattr(node, "mutation_type", None) or "",
        "summary": node.mutation_plan,
        "mutation_plan": node.mutation_plan,
        "metric": node.metric,
        "reward": node.reward,
        "validity": "valid" if node.is_valid else "invalid",
        "error_buckets": dict(node.error_buckets or {}),
        "branch_id": node.branch_id,
    }
