"""Stage 3 — self-distill from the trained model (rejection sampling).

Same shape as Stage 2 but typically points at the 30B endpoint, samples
N>1 per row at higher temperature, and keeps every sample whose boxed
answer matches kaggle_answer. Disabled by default — opt-in via
``enabled: true``.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .teacher_distill import _call_vllm, _extract_boxed, _render_prompt


def run(stage_cfg: dict[str, Any], stage1_out: Path,
        prompt_templates: dict[str, str], log) -> dict[str, int]:
    if not stage_cfg.get("enabled", False):
        log("  stage_3_self_distill: disabled")
        return {"skipped": True}

    endpoint = stage_cfg["endpoint"]
    base_url = endpoint["base_url"]
    model    = endpoint["model"]
    sampling = stage_cfg.get("sampling", {})
    # pass@k semantics — same shape as stage_2_teacher_distill. Default
    # keep_policy here is "all" so rejection sampling collects every hit
    # (a row may have several distinct correct CoTs worth keeping).
    k = int(stage_cfg.get("k",
            sampling.get("n_samples_per_row", 4)))
    keep_policy = stage_cfg.get("keep_policy", "all")
    if keep_policy not in ("all", "first_correct"):
        raise ValueError(f"unknown keep_policy {keep_policy!r}")
    parallelism = int(stage_cfg.get("parallelism", 4))
    template_key = stage_cfg["prompt_template_key"]
    template = prompt_templates[template_key]
    prefill_key = stage_cfg.get("prefill_template_key")
    prefill_template = prompt_templates[prefill_key] if prefill_key else None
    timeout = float(stage_cfg.get("timeout_seconds", 180.0))
    out_path = Path(stage_cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in stage1_out.read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("status") == "ok"]
    log(f"  stage_3: {len(rows)} rows × pass@k={k} keep={keep_policy}"
        f"  prefill={'on' if prefill_template else 'off'}")

    counts = {"calls": 0, "accepted": 0, "rejected": 0,
              "http_error": 0, "kept": 0}

    def _one_call(row: dict[str, Any], j: int) -> dict[str, Any]:
        try:
            rendered = _render_prompt(template, row)
            prefill = (_render_prompt(prefill_template, row)
                       if prefill_template else None)
            continuation = _call_vllm(base_url, model, rendered, sampling,
                                      timeout, assistant_prefill=prefill)
        except urllib.error.URLError as exc:
            return {"row_id": row["row_id"], "sample_k": j,
                    "error": f"http: {exc!r}", "_bucket": "http_error"}
        completion = (prefill + continuation) if prefill else continuation
        boxed = _extract_boxed(completion)
        accepted = bool(boxed) and boxed == row["kaggle_answer"]
        return {
            "row_id": row["row_id"],
            "sample_k": j,
            "prompt": row["prompt"],
            "kaggle_answer": row["kaggle_answer"],
            "completion": completion,
            "boxed": boxed,
            "accepted": accepted,
            "_bucket": "accepted" if accepted else "rejected",
        }

    def _process_row(row: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for j in range(k):
            res = _one_call(row, j)
            results.append(res)
            if keep_policy == "first_correct" and res.get("accepted"):
                break
        return results

    if keep_policy == "all":
        total_calls_planned = len(rows) * k
        unit_label = "calls"
    else:
        total_calls_planned = len(rows)
        unit_label = "rows"

    with out_path.open("w") as out, cf.ThreadPoolExecutor(parallelism) as ex:
        if keep_policy == "all":
            futures = [ex.submit(_one_call, row, j)
                       for row in rows for j in range(k)]
        else:
            futures = [ex.submit(_process_row, row) for row in rows]

        bar = tqdm(
            total=total_calls_planned, unit=unit_label, smoothing=0.05,
            mininterval=2.0, maxinterval=30.0,
            desc="stage_3", file=sys.stderr,
            dynamic_ncols=True,
        )
        log_every_n = max(1, total_calls_planned // 100)
        for fut in cf.as_completed(futures):
            result_bag = fut.result()
            if not isinstance(result_bag, list):
                result_bag = [result_bag]
            for res in result_bag:
                counts["calls"] += 1
                bucket = res.pop("_bucket", "rejected")
                counts[bucket] = counts.get(bucket, 0) + 1
                # Self-distill always drops non-accepted samples (rejection
                # sampling is the whole point). first_correct additionally
                # stops dispatch on the first hit per row.
                if not res.get("accepted"):
                    continue
                out.write(json.dumps(res) + "\n")
                out.flush()
                counts["kept"] += 1
                if counts["calls"] % log_every_n == 0:
                    log(f"    progress {counts['calls']}/{total_calls_planned} "
                        f"accepted={counts['accepted']} "
                        f"rejected={counts['rejected']} "
                        f"kept={counts['kept']}")
            bar.update(1)
            bar.set_postfix(
                accepted=counts["accepted"],
                kept=counts["kept"],
                refresh=False,
            )
        bar.close()

    rows_with_hit = len({r["row_id"] for r in
                         (json.loads(l) for l in
                          out_path.read_text().splitlines() if l.strip())
                         if r.get("accepted")})
    row_pass_rate = rows_with_hit / max(1, len(rows))
    log(f"  stage_3: rows_with_hit={rows_with_hit}/{len(rows)} "
        f"(pass@{k}={row_pass_rate:.3f})  "
        f"per_call_accepted={counts['accepted']}/{counts['calls']}  "
        f"rejected={counts['rejected']}  http_err={counts['http_error']} "
        f"kept={counts['kept']}")
    log(f"  wrote {out_path}")
    return counts
