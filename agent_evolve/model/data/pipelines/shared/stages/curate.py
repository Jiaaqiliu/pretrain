"""Stage 5 — curate (format-filter + dedup) the accepted Stage 2/3 outputs
into a single domain JSONL ready for downstream mixing.

Output rows match the canonical legacy training-data schema produced by
huikang's 0.85-LB Kaggle pipeline:

    {
      "id":              "<kaggle_row>-<source>-<sample_k>",
      "type":            "<domain>",
      "prompt_rendered": "<chat-template-wrapped raw user prompt>",
      "completion":      "<CoT>...</think>\\n\\boxed{<answer>}<|im_end|>",
      "source":          "teacher" | "self",  // extra, harmless to trainer
    }

That alignment makes the curated JSONL **mix-ready** with
``default_14718.jsonl``: train_unsloth.py reads ``prompt_rendered`` +
``completion`` directly, the entire CoT lives inside the model's
``<think>`` block, and the boxed answer sits AFTER ``</think>`` like
every legacy row.

Format alignment is gated on the ``format_alignment`` config block:

    stage_5_curate:
      format_alignment:
        tokenizer: /fsx/models/Nemotron-3-Nano-30B-A3B-unsloth
        domain_tag: bits

When the block is absent (legacy / smoke-test mode), curate falls back
to the plain ``{id, source, prompt, completion}`` shape.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


# Used by the format_alignment path to split the curated CoT into a
# pre-think narrative + post-think boxed answer. See
# bits/prompt_templates.yaml::teacher_v2 — the 3-section structure is
# stable across all bits curated rows.
_FINAL_ANSWER_HEADER = re.compile(r"\*\*3\. Final answer\.\*\*")


def _reshape_completion_for_legacy(completion: str) -> str:
    """Pipeline form (no </think>) → legacy form (closes think before boxed)."""
    m = _FINAL_ANSWER_HEADER.search(completion)
    if m is None:
        bm = re.search(r"\\boxed\{[^}]*\}", completion)
        if bm is None:
            return completion + "\n</think><|im_end|>"
        split_pos = bm.start()
    else:
        split_pos = m.start()
    think = completion[:split_pos].rstrip()
    answer = completion[split_pos:].strip()
    return f"{think}\n</think>\n{answer}<|im_end|>"


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


def _load_audit_pass_set(audit_path: Path | None) -> set[tuple[str, str, int]] | None:
    """Build a set of (source, row_id, sample_k) tuples that Opus marked
    as ``pass``. Used by ``require_audit_pass`` to intersect with the raw
    Stage 2 / 3 output. Returns None if audit not available — callers
    treat that as "filter disabled."
    """
    if audit_path is None or not audit_path.exists():
        return None
    keys: set[tuple[str, str, int]] = set()
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        v = json.loads(line)
        if not v.get("pass"):
            continue
        # Stage 4 verdicts carry source / row_id / sample_k (legacy verdicts
        # may lack sample_k → default to 0 to mirror the curate dedup key).
        keys.add(
            (v.get("source") or "teacher",
             v["row_id"],
             v.get("sample_k") if v.get("sample_k") is not None else 0)
        )
    return keys


def run(stage_cfg: dict[str, Any], stage_inputs: list[Path],
        log, audit_path: Path | None = None) -> dict[str, Any]:
    filters = stage_cfg.get("format_filters", {})
    dedup_cfg = stage_cfg.get("dedup", {"by": "row_id", "keep": "first"})
    out_template = stage_cfg["out_path"]

    # Format alignment (optional). When present, rows are emitted in the
    # legacy default_14718 schema so the curated set is mix-ready with
    # the production training JSONL.
    align_cfg = stage_cfg.get("format_alignment")
    apply_chat_template_fn = None
    domain_tag = None
    if align_cfg:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            align_cfg["tokenizer"], trust_remote_code=True,
        )
        def apply_chat_template_fn(text: str) -> str:
            return tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )
        domain_tag = align_cfg.get("domain_tag", "unknown")
        log(f"  stage_5: format_alignment ON "
            f"(tokenizer={align_cfg['tokenizer']}, type={domain_tag})")

    require_audit_pass = bool(filters.get("require_audit_pass", False))
    audit_set: set[tuple[str, str, int]] | None = None
    if require_audit_pass:
        audit_set = _load_audit_pass_set(audit_path)
        if audit_set is None:
            raise RuntimeError(
                "stage_5: format_filters.require_audit_pass=true but no "
                "stage_4 audit JSONL was provided / found. Re-run "
                "Stage 4 first or unset the filter."
            )
        log(f"  stage_5: audit-gating enabled — keeping only rows in "
            f"the {len(audit_set)} Opus-passed verdicts")

    accepted: dict[str, dict[str, Any]] = {}
    counts = {"scanned": 0, "format_rejected": 0, "duplicate_dropped": 0,
              "audit_rejected": 0,
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
            if audit_set is not None:
                audit_key = (
                    source, r["row_id"],
                    r.get("sample_k") if r.get("sample_k") is not None else 0,
                )
                if audit_key not in audit_set:
                    counts["audit_rejected"] += 1
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
            if apply_chat_template_fn is not None:
                # Aligned schema: matches default_14718 verbatim.
                # `source` is kept as an extra field for human inspection;
                # train_unsloth.py only reads prompt_rendered + completion.
                accepted[key] = {
                    "id":              curated_id,
                    "type":            domain_tag,
                    "prompt_rendered": apply_chat_template_fn(r["prompt"]),
                    "completion":      _reshape_completion_for_legacy(r["completion"]),
                    "source":          source,
                }
            else:
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
        f"audit_rejected={counts['audit_rejected']} "
        f"duplicate_dropped={counts['duplicate_dropped']} hash={digest}")
    log(f"  wrote {out_path}")
    return {**counts, "out_path": str(out_path), "hash": digest}
