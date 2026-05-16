"""Stage 5 — curate (format-filter + dedup) the accepted Stage 2/3 outputs
into a single domain JSONL ready for downstream mixing.

Output rows match the canonical training-data schema (one JSON object per
line with ``prompt`` and ``completion`` keys), so the existing tokenizer
in train_unsloth.py picks them up unchanged. Each row also carries a
``source: "teacher" | "self"`` field so the planner / data-mix step can
weight the two generators independently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _format_ok(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    cot = row.get("completion", "")
    if filters.get("must_end_with_boxed", True):
        if "\\boxed{" not in cot:
            return False
    max_chars = int(filters.get("max_chars", 1_000_000))
    min_chars = int(filters.get("min_chars", 0))
    if not (min_chars <= len(cot) <= max_chars):
        return False
    return True


def _source_tag(path: Path) -> str:
    """Derive a short source tag from the input path for multi-pass id-ing.

    Stage 2 writes ``..._teacher.jsonl`` / ``stage2.jsonl``; Stage 3 writes
    ``..._self.jsonl`` / ``stage3.jsonl``. We use the filename stem so a
    ``(row_id, sample_k)`` from teacher doesn't collide with the same
    ``(row_id, sample_k)`` from self-distill.
    """
    stem = path.stem.lower()
    if "teacher" in stem or stem.endswith("stage2"):
        return "teacher"
    if "self" in stem or stem.endswith("stage3"):
        return "self"
    return stem


def run(stage_cfg: dict[str, Any], stage_inputs: list[Path],
        log) -> dict[str, Any]:
    filters = stage_cfg.get("format_filters", {})
    dedup_cfg = stage_cfg.get("dedup", {"by": "row_id", "keep": "first"})
    out_template = stage_cfg["out_path"]

    accepted: dict[str, dict[str, Any]] = {}
    counts = {"scanned": 0, "format_rejected": 0, "duplicate_dropped": 0,
              "kept": 0, "kept_teacher": 0, "kept_self": 0}

    for path in stage_inputs:
        if not path.exists():
            continue
        source = _source_tag(path)
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            counts["scanned"] += 1
            if not r.get("accepted"):
                continue
            if not _format_ok(r, filters):
                counts["format_rejected"] += 1
                continue
            dedup_by = dedup_cfg.get("by", "row_id")
            if dedup_by == "row_id":
                key = r["row_id"]
            elif dedup_by == "sample":
                # Multi-pass: keep one row per (source, row_id, sample_k) so
                # teacher and self-distill samples for the same Kaggle row
                # don't collide. Rows missing sample_k (legacy single-pass)
                # use sample_k=0.
                key = (source, r["row_id"], r.get("sample_k", 0))
            else:
                key = json.dumps(r, sort_keys=True)
            if key in accepted:
                counts["duplicate_dropped"] += 1
                continue
            row_id = r["row_id"]
            sample_k = r.get("sample_k")
            curated_id = (
                f"{row_id}-{source}-{sample_k}"
                if sample_k is not None and dedup_by == "sample"
                else row_id
            )
            accepted[key] = {
                "id": curated_id,
                "source": source,
                "prompt": r["prompt"],
                "completion": r["completion"],
            }
            counts["kept"] += 1
            counts[f"kept_{source}"] = counts.get(f"kept_{source}", 0) + 1

    # Deterministic order so the hash is stable
    rows_sorted = sorted(accepted.values(), key=lambda r: r["id"])
    payload = "\n".join(json.dumps(r) for r in rows_sorted) + "\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    out_path = Path(out_template.replace("${hash}", digest))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload)

    log(f"  stage_5: scanned={counts['scanned']} kept={counts['kept']} "
        f"(teacher={counts['kept_teacher']} self={counts['kept_self']}) "
        f"format_rejected={counts['format_rejected']} "
        f"duplicate_dropped={counts['duplicate_dropped']} hash={digest}")
    log(f"  wrote {out_path}")
    return {**counts, "out_path": str(out_path), "hash": digest}
