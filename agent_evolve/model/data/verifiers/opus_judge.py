"""Verify dev-set labels by cross-checking with an external judge (Claude Opus 4.6 via Bedrock).

Use case: we just built `balanced_dev600`. Before trusting scores on it, confirm
the stored `answer` for each row is actually correct. A strong external model
that has never seen our training data acts as an independent oracle: if Opus
agrees with the stored answer on a row, the label is very likely right; if
Opus disagrees on many rows in a domain, the dev set may have systematic
labeling errors.

Input:  a CSV with columns id, prompt, answer, domain (the shape produced
        by data/scripts/build_dev_set).
Output: per-row judgments + per-domain agreement rates + a list of
        rows that need human review.

Usage:
    python -m agent_evolve.model.data.verifiers.opus_judge \
        --input  runs/.../eval/balanced_dev600.csv \
        --output runs/.../eval/balanced_dev600.verified.jsonl \
        --model  anthropic.claude-opus-4-6-v1 \
        --limit  600 \
        --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, "/fsx/zzsamshi/a-evolve")

from agent_evolve.benchmarks.nemo_reasoner import extract_final_answer, verify  # noqa: E402
from agent_evolve.llm.base import LLMMessage  # noqa: E402
from agent_evolve.llm.bedrock import BedrockProvider  # noqa: E402

logger = logging.getLogger(__name__)

# Mirror the exact prompt suffix the Kaggle host + our eval append.
EVAL_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


@dataclass
class Row:
    id: str
    prompt: str
    answer: str
    domain: str
    source: str = ""


def load_rows(path: Path, limit: int | None = None) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Row(
                    id=r["id"],
                    prompt=r["prompt"],
                    answer=r["answer"],
                    domain=r.get("domain", "unknown"),
                    source=r.get("source", ""),
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def judge_row(
    provider: BedrockProvider, row: Row, max_tokens: int, temperature: float
) -> dict[str, Any]:
    """Ask the external judge for its answer to row; return a verdict dict."""
    messages = [
        LLMMessage(role="user", content=row.prompt + EVAL_SUFFIX),
    ]
    try:
        resp = provider.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = resp.content
        pred = extract_final_answer(raw)
        agrees = verify(row.answer, pred) if pred else False
        return {
            "id": row.id,
            "domain": row.domain,
            "source": row.source,
            "stored_answer": row.answer,
            "judge_prediction": pred,
            "agrees": agrees,
            "judge_raw_len": len(raw),
            "judge_raw_tail": raw[-400:] if raw else "",
            "input_tokens": resp.usage.get("input_tokens", 0) if resp.usage else 0,
            "output_tokens": resp.usage.get("output_tokens", 0) if resp.usage else 0,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("judge call failed for id=%s", row.id)
        return {
            "id": row.id,
            "domain": row.domain,
            "source": row.source,
            "stored_answer": row.answer,
            "judge_prediction": None,
            "agrees": False,
            "judge_raw_len": 0,
            "judge_raw_tail": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "error": repr(exc),
        }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Dev CSV (id,prompt,answer,domain,source).")
    ap.add_argument("--output", required=True, type=Path, help="Output JSONL path for verdicts.")
    ap.add_argument(
        "--model",
        default="anthropic.claude-opus-4-6-v1",
        help="Bedrock model ID (default: Opus 4.6).",
    )
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="Cap rows for a dry run.")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent Bedrock requests.")
    args = ap.parse_args()

    rows = load_rows(args.input, limit=args.limit)
    logger.info("loaded %d rows from %s", len(rows), args.input)

    provider = BedrockProvider(model_id=args.model, region=args.region)
    logger.info("using judge model=%s region=%s", args.model, args.region)

    verdicts: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Stream to output so a crash doesn't lose completed judgments.
    with open(args.output, "w") as f_out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(judge_row, provider, r, args.max_tokens, args.temperature): r for r in rows}
        for i, fut in enumerate(as_completed(futs)):
            v = fut.result()
            verdicts.append(v)
            f_out.write(json.dumps(v) + "\n")
            f_out.flush()
            if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                logger.info(
                    "  progress %d/%d  last id=%s agrees=%s",
                    i + 1, len(rows), v["id"], v["agrees"],
                )

    # Summary
    n = len(verdicts)
    n_errors = sum(1 for v in verdicts if v["error"])
    n_agree = sum(1 for v in verdicts if v["agrees"])
    n_disagree = n - n_agree - n_errors

    print(f"\n=== verification summary ===")
    print(f"total:        {n}")
    print(f"judge errors: {n_errors}")
    print(f"agree:        {n_agree} ({100*n_agree/max(1,n):.1f}%)")
    print(f"disagree:     {n_disagree} ({100*n_disagree/max(1,n):.1f}%)")

    # Per-domain breakdown
    per_dom: dict[str, dict[str, int]] = {}
    for v in verdicts:
        d = v["domain"]
        b = per_dom.setdefault(d, {"n": 0, "agree": 0, "error": 0, "held_out_agree": 0, "held_out_n": 0, "train_agree": 0, "train_n": 0})
        b["n"] += 1
        if v["error"]:
            b["error"] += 1
        elif v["agrees"]:
            b["agree"] += 1
        if v["source"] == "held_out":
            b["held_out_n"] += 1
            if v["agrees"]:
                b["held_out_agree"] += 1
        elif v["source"] == "from_training":
            b["train_n"] += 1
            if v["agrees"]:
                b["train_agree"] += 1

    print(f"\n=== per-domain agreement (higher = more trustworthy labels) ===")
    print(f"{'domain':12s} {'n':>4s} {'agree':>7s} {'rate':>6s}   {'held_out':>14s}   {'from_train':>14s}")
    for d in sorted(per_dom):
        b = per_dom[d]
        ho = f"{b['held_out_agree']}/{b['held_out_n']}" if b['held_out_n'] else "-"
        tr = f"{b['train_agree']}/{b['train_n']}" if b['train_n'] else "-"
        print(
            f"{d:12s} {b['n']:>4d} {b['agree']:>7d} {100*b['agree']/max(1,b['n']):>5.1f}%   "
            f"{ho:>14s}   {tr:>14s}"
        )

    # Flag suspected-bad rows
    bad = [v for v in verdicts if not v["agrees"] and not v["error"]]
    if bad:
        print(f"\n=== top 5 disagreements (possible label errors) ===")
        for v in bad[:5]:
            print(
                f"  id={v['id']} dom={v['domain']} stored={v['stored_answer']!r} "
                f"judge_pred={v['judge_prediction']!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
