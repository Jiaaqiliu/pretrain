"""Stage 4 — Opus process supervision.

Reads accepted rows from BOTH Stage 2 (teacher) and Stage 3 (self-distill)
and asks Opus 4.6 whether the CoT narrative actually derives the boxed
answer (per the domain-specific rubric). Each verdict carries a
``source`` tag (``"teacher"`` / ``"self"``) so Stage 5 / planner can see
which generator produced which row. Used as a quality gate: if pass rate
falls below ``pass_threshold`` the runner halts.

Modes:
  * ``full``   — judge every accepted row from both sources. Default.
  * ``sample`` — judge a random ``sample_size`` per source.
  * ``rewrite`` — ask Opus to rewrite low-quality CoTs. Out of scope for v1.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _try_anthropic_client():
    try:
        import anthropic  # noqa: F401
        return anthropic
    except ImportError:
        return None


def _judge(rubric: str, row: dict[str, Any], model: str,
           timeout: float, anthropic_mod) -> dict[str, Any]:
    user_prompt = (
        rubric
        .replace("{examples_block}", row.get("prompt", ""))
        .replace("{question}", row.get("prompt", ""))
        .replace("{cot}", row.get("completion", ""))
        .replace("{boxed}", row.get("boxed", "") or "")
        .replace("{kaggle_answer}", row.get("kaggle_answer", ""))
    )
    client = anthropic_mod.AnthropicBedrock(
        aws_region=os.environ.get("AWS_REGION", "us-west-2"),
    )
    msg = client.messages.create(
        model=model, max_tokens=512, timeout=timeout,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = msg.content[0].text if msg.content else ""
    # Pull the JSON object out of the response (Opus may pad with prose)
    m = re.search(r"\{[^{}]*\}", text, re.S)
    parsed = {}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {
        "row_id": row["row_id"],
        "sample_k": row.get("sample_k"),
        "source": row.get("_source"),
        "raw": text,
        "parsed": parsed,
        "pass": bool(parsed.get("overall_pass", False)),
    }


def _load_accepted(path: Path | None, source: str) -> list[dict[str, Any]]:
    """Read accepted rows from a stage-2/3 output and tag each with ``source``.

    Tag is stored under the leading-underscore key ``_source`` so it doesn't
    collide with the schema seen by ``_judge`` and is preserved on the
    verdict written to disk.
    """
    if path is None or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("accepted"):
            continue
        r["_source"] = source
        out.append(r)
    return out


def run(stage_cfg: dict[str, Any], stage_inputs: dict[str, Path | None],
        prompt_blocks: dict[str, str], log) -> dict[str, Any]:
    """``stage_inputs`` is ``{"teacher": <path|None>, "self": <path|None>}``.

    Either source may be missing (Stage 3 disabled, prior run skipped, etc.).
    Each accepted row is judged and tagged with its source on the verdict.
    """
    if not stage_cfg.get("enabled", True):
        log("  stage_4_opus_supervision: disabled")
        return {"skipped": True}

    anthropic_mod = _try_anthropic_client()
    if anthropic_mod is None:
        log("  stage_4: anthropic SDK not installed; skipping (install: "
            "pip install anthropic[bedrock])")
        return {"skipped": True, "reason": "sdk_missing"}

    mode = stage_cfg.get("mode", "full")
    sample_size = int(stage_cfg.get("sample_size", 200))
    seed = int(stage_cfg.get("sample_seed", 42))
    model = stage_cfg.get("model", "claude-opus-4-6")
    rubric_key = stage_cfg["rubric_key"]
    rubric = prompt_blocks[rubric_key]
    pass_threshold = float(stage_cfg.get("pass_threshold", 0.90))
    parallelism = int(stage_cfg.get("api", {}).get("max_concurrency", 8))
    timeout = float(stage_cfg.get("api", {}).get("timeout_seconds", 120.0))
    out_path = Path(stage_cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    teacher_rows = _load_accepted(stage_inputs.get("teacher"), "teacher")
    self_rows    = _load_accepted(stage_inputs.get("self"),    "self")
    if mode == "sample":
        rng = random.Random(seed)
        rng.shuffle(teacher_rows)
        rng.shuffle(self_rows)
        teacher_rows = teacher_rows[:sample_size]
        self_rows    = self_rows[:sample_size]
    rows = teacher_rows + self_rows
    log(f"  stage_4: judging {len(rows)} rows ({mode}) "
        f"teacher={len(teacher_rows)} self={len(self_rows)}")

    counts = {
        "judged": len(rows), "pass": 0,
        "teacher_judged": len(teacher_rows), "teacher_pass": 0,
        "self_judged":    len(self_rows),    "self_pass":    0,
    }

    def _go(r):
        return _judge(rubric, r, model, timeout, anthropic_mod)

    log_every_n = max(1, len(rows) // 50)
    n_done = 0
    with out_path.open("w") as out, cf.ThreadPoolExecutor(parallelism) as ex:
        bar = tqdm(
            total=len(rows), unit="rows", smoothing=0.05,
            mininterval=2.0, maxinterval=30.0,
            desc="stage_4", file=sys.stderr,
            dynamic_ncols=True,
        )
        for verdict in ex.map(_go, rows):
            if verdict.get("pass"):
                counts["pass"] += 1
                src_key = f"{verdict.get('source', 'teacher')}_pass"
                counts[src_key] = counts.get(src_key, 0) + 1
            out.write(json.dumps(verdict) + "\n")
            out.flush()
            n_done += 1
            bar.update(1)
            bar.set_postfix_str(
                f"pass={counts['pass']} "
                f"T={counts['teacher_pass']} S={counts['self_pass']}",
                refresh=False,
            )
            if n_done % log_every_n == 0:
                log(f"    progress {n_done}/{len(rows)} pass={counts['pass']} "
                    f"(teacher={counts['teacher_pass']}/{counts['teacher_judged']} "
                    f"self={counts['self_pass']}/{counts['self_judged']})")
        bar.close()

    pass_rate = counts["pass"] / max(1, counts["judged"])
    teacher_pr = counts["teacher_pass"] / max(1, counts["teacher_judged"])
    self_pr    = counts["self_pass"]    / max(1, counts["self_judged"])
    log(f"  stage_4: pass={counts['pass']}/{counts['judged']}  "
        f"pass_rate={pass_rate:.3f}  "
        f"teacher={counts['teacher_pass']}/{counts['teacher_judged']}"
        f" ({teacher_pr:.3f})  "
        f"self={counts['self_pass']}/{counts['self_judged']} ({self_pr:.3f})")
    if pass_rate < pass_threshold:
        raise RuntimeError(
            f"stage_4 pass rate {pass_rate:.3f} < threshold "
            f"{pass_threshold:.3f}; halt and revisit prompt template"
        )
    return counts
