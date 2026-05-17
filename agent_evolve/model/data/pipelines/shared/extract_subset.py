"""Per-domain subset extraction + dev-row decontamination + format alignment.

Three modes drive the trainer's data-ablation flow:

  baseline      filter ``default_14718.jsonl`` (or any train JSONL) down
                to a single domain so we can train a "baseline" arm on
                only that domain's original rows.
  decontaminate read a curated JSONL produced by dw-pipeline-launch and
                drop any row whose Kaggle row_id appears in the dev split
                (``balanced_dev726.csv``). Curated row ids look like
                ``<kaggle_id>-(teacher|self)-<sample_k>``; we strip the
                suffix to recover the kaggle_id.
  reformat      align a curated JSONL with the legacy default_14718
                schema (``{id, type, prompt_rendered, completion}``)
                so it is mix-ready: chat-template the raw ``prompt``
                into ``prompt_rendered`` and split the completion at
                the final-answer header so the ``\\boxed{…}`` lands
                AFTER ``</think>`` exactly like huikang's CoTs.

All modes write a sibling ``provenance.json`` recording row counts,
sha256, and any dropped count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from .stages.witness_search import DOMAIN_HEURISTIC


# Curated id format: "<kaggle_id>-<source>-<sample_k>" (see curate.py).
# The kaggle_id is whatever was the original ``row_id`` in stage1.jsonl —
# typically a hex hash but we don't constrain its alphabet here.
_CURATED_ID_RE = re.compile(r"^(.+)-(?:teacher|self)-\d+$")

# The curated CoT body uses a fixed three-section template (see
# bits/prompt_templates.yaml::teacher_v2): section 3 is the final-answer
# section that always immediately precedes the \boxed{...}. Splitting at
# the section header lets us put the entire CoT inside <think> and the
# boxed answer outside, matching legacy default_14718 rows where
# `</think>\n\boxed{...}<|im_end|>` is the canonical tail.
_FINAL_ANSWER_HEADER = re.compile(r"\*\*3\. Final answer\.\*\*")


def _row_prompt(r: dict[str, Any]) -> str:
    """Extract the prompt-shaped text from a training row.

    The canonical Kaggle JSONL uses ``prompt_rendered`` (chat-template
    wrapped); the curated JSONL we emit uses ``prompt``. Both work for
    the heuristic — return whichever is present.
    """
    return r.get("prompt_rendered") or r.get("prompt") or ""


def _strip_curated_suffix(curated_id: str) -> str:
    """Recover the kaggle_id from a curated row's compound id.

    Falls back to the id as-is when no suffix matches (legacy
    single-pass curated files).
    """
    m = _CURATED_ID_RE.match(curated_id)
    return m.group(1) if m else curated_id


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _write_provenance(out_dir: Path, payload: dict[str, Any]) -> Path:
    p = out_dir / "provenance.json"
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def cmd_baseline(args: argparse.Namespace) -> int:
    src = Path(args.src)
    domain = args.domain
    if domain not in DOMAIN_HEURISTIC:
        print(f"unknown domain {domain!r}; known: {sorted(DOMAIN_HEURISTIC)}",
              file=sys.stderr)
        return 2
    predicate = DOMAIN_HEURISTIC[domain]

    # Stream rows once, write to a temp path, then rename into the
    # content-addressed location. Hash is computed on the post-filter
    # bytes for determinism.
    out_dir_root = Path(args.out_dir)
    tmp_path = out_dir_root.parent / f".{out_dir_root.name}.tmp.jsonl"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()

    rows_in = 0
    rows_out = 0
    with src.open() as fin, tmp_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rows_in += 1
            r = json.loads(line)
            if predicate(_row_prompt(r)):
                fout.write(line)
                rows_out += 1

    digest = _sha256_short(tmp_path)
    final_dir = out_dir_root / digest
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"baseline_{domain}.jsonl"
    tmp_path.replace(final_path)

    _write_provenance(final_dir, {
        "mode": "baseline",
        "src": str(src),
        "out": str(final_path),
        "domain": domain,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped": rows_in - rows_out,
        "sha256": digest,
    })
    print(json.dumps({
        "ok": True, "mode": "baseline", "domain": domain,
        "out": str(final_path), "rows_in": rows_in, "rows_out": rows_out,
        "sha256": digest,
    }))
    return 0


def _wrap_user_prompt(prompt_text: str, tokenizer_path: str) -> str:
    """Apply the model's chat template to a raw user prompt.

    Produces the exact byte sequence
    ``<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<text><|im_end|>\n<|im_start|>assistant\n<think>\n``
    that legacy default_14718 rows already store as ``prompt_rendered``.

    Lazy-imported because the rest of this CLI is dependency-free.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )


def _reshape_completion(completion: str) -> str:
    """Convert a pipeline-emitted completion into legacy think-block form.

    Pipeline form (one assistant turn, no </think>):
        Looking at the worked examples...
        ...verification...
        **3. Final answer.**
        \\boxed{<answer>}

    Legacy form (entire CoT inside <think>, boxed answer post-think):
        Looking at the worked examples...
        ...verification...
        </think>
        \\boxed{<answer>}<|im_end|>

    Splits at the ``**3. Final answer.**`` header — verified to be
    present + unique in all 1019 audit-passed bits CoTs.
    """
    m = _FINAL_ANSWER_HEADER.search(completion)
    if m is None:
        # No structured section header — fall back to splitting right
        # before the LAST \boxed{…}. Catches future curated rows whose
        # template doesn't use the exact section string.
        bm = re.search(r"\\boxed\{[^}]*\}", completion)
        if bm is None:
            raise RuntimeError(
                "completion has neither '**3. Final answer.**' nor "
                "\\boxed{…}; cannot reshape"
            )
        split_pos = bm.start()
    else:
        split_pos = m.start()
    think = completion[:split_pos].rstrip()
    answer = completion[split_pos:].strip()
    return f"{think}\n</think>\n{answer}<|im_end|>"


def cmd_reformat(args: argparse.Namespace) -> int:
    """Convert a curated JSONL into the legacy default_14718 schema.

    Input row:  {"id": "<kaggle>-teacher-0", "source": "teacher",
                 "prompt": "<raw>", "completion": "<no </think>>"}
    Output row: {"id": "<kaggle>-teacher-0", "type": "<domain>",
                 "prompt_rendered": "<chat-template-wrapped>",
                 "completion": "<...</think>\\n\\boxed{...}<|im_end|>"}

    The output is byte-equivalent in shape to default_14718 rows, so
    the trainer's `prompt_rendered` reader works without modification
    and the curated set is ready to mix into the production training
    JSONL.
    """
    src = Path(args.src)
    domain = args.domain
    tokenizer_path = args.tokenizer

    # Load tokenizer once (it's the slow path).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    out_dir_root = Path(args.out_dir)
    tmp_path = out_dir_root.parent / f".{out_dir_root.name}.tmp.jsonl"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()

    rows_in = 0
    rows_out = 0
    rows_skipped = 0
    with src.open() as fin, tmp_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rows_in += 1
            r = json.loads(line)
            if "prompt" not in r or "completion" not in r:
                rows_skipped += 1
                continue
            try:
                wrapped = tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                reshaped = _reshape_completion(r["completion"])
            except Exception:
                rows_skipped += 1
                continue
            out_row = {
                "id": r.get("id", ""),
                "type": domain,
                "prompt_rendered": wrapped,
                "completion": reshaped,
            }
            fout.write(json.dumps(out_row) + "\n")
            rows_out += 1

    digest = _sha256_short(tmp_path)
    final_dir = out_dir_root / digest
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"curated_aligned_{domain}.jsonl"
    tmp_path.replace(final_path)

    _write_provenance(final_dir, {
        "mode": "reformat",
        "src": str(src),
        "out": str(final_path),
        "domain": domain,
        "tokenizer": tokenizer_path,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_skipped": rows_skipped,
        "sha256": digest,
    })
    print(json.dumps({
        "ok": True, "mode": "reformat", "domain": domain,
        "out": str(final_path),
        "rows_in": rows_in, "rows_out": rows_out,
        "rows_skipped": rows_skipped, "sha256": digest,
    }))
    return 0


def cmd_decontaminate(args: argparse.Namespace) -> int:
    src = Path(args.src)
    dev = Path(args.dev)

    csv.field_size_limit(sys.maxsize)
    dev_ids: set[str] = set()
    with dev.open(newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            print(f"dev CSV missing 'id' column: {dev}", file=sys.stderr)
            return 2
        for row in reader:
            dev_ids.add(row["id"])
    if not dev_ids:
        print(f"empty dev id set from {dev}", file=sys.stderr)
        return 2

    out_dir_root = Path(args.out_dir)
    tmp_path = out_dir_root.parent / f".{out_dir_root.name}.tmp.jsonl"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()

    rows_in = 0
    rows_out = 0
    with src.open() as fin, tmp_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rows_in += 1
            r = json.loads(line)
            curated_id = r.get("id", "")
            kaggle_id = _strip_curated_suffix(curated_id)
            if kaggle_id in dev_ids:
                continue
            fout.write(line)
            rows_out += 1

    digest = _sha256_short(tmp_path)
    final_dir = out_dir_root / digest
    final_dir.mkdir(parents=True, exist_ok=True)
    # Domain inferred from src filename suffix (curated files are
    # ``<domain>_distilled.jsonl`` per curate.py); fall back to "curated".
    base = src.stem.replace("_distilled", "")
    final_name = f"curated_clean_{base}.jsonl"
    final_path = final_dir / final_name
    tmp_path.replace(final_path)

    _write_provenance(final_dir, {
        "mode": "decontaminate",
        "src": str(src),
        "dev": str(dev),
        "out": str(final_path),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "decontam_dropped": rows_in - rows_out,
        "dev_id_count": len(dev_ids),
        "sha256": digest,
    })
    print(json.dumps({
        "ok": True, "mode": "decontaminate",
        "out": str(final_path), "rows_in": rows_in, "rows_out": rows_out,
        "decontam_dropped": rows_in - rows_out, "sha256": digest,
    }))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("baseline", help="filter a train JSONL by domain")
    sb.add_argument("--src", required=True,
                    help="path to default_14718.jsonl (or compatible)")
    sb.add_argument("--domain", required=True,
                    choices=sorted(DOMAIN_HEURISTIC.keys()))
    sb.add_argument("--out-dir", required=True,
                    help="root dir; output lands at <out-dir>/<sha12>/baseline_<domain>.jsonl")
    sb.set_defaults(func=cmd_baseline)

    sd = sub.add_parser("decontaminate",
                        help="drop curated rows whose kaggle_id is in the dev set")
    sd.add_argument("--src", required=True,
                    help="curated JSONL (e.g. <hash>/<domain>_distilled.jsonl)")
    sd.add_argument("--dev", required=True,
                    help="path to balanced_dev726.csv (or compatible)")
    sd.add_argument("--out-dir", required=True,
                    help="root dir; output lands at <out-dir>/<sha12>/curated_clean_<domain>.jsonl")
    sd.set_defaults(func=cmd_decontaminate)

    sr = sub.add_parser("reformat",
                        help="align a curated JSONL with default_14718 schema "
                             "(prompt_rendered + post-</think> boxed answer)")
    sr.add_argument("--src", required=True,
                    help="curated JSONL emitted by curate.py "
                         "(may already be decontaminated)")
    sr.add_argument("--domain", required=True,
                    choices=sorted(DOMAIN_HEURISTIC.keys()),
                    help="written into the `type` field of each output row")
    sr.add_argument("--tokenizer", default="/fsx/models/Nemotron-3-Nano-30B-A3B-unsloth",
                    help="HF tokenizer path used for chat-template wrapping")
    sr.add_argument("--out-dir", required=True,
                    help="root dir; output lands at "
                         "<out-dir>/<sha12>/curated_aligned_<domain>.jsonl")
    sr.set_defaults(func=cmd_reformat)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
