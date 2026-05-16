"""Stage 2 — teacher distillation.

Reads Stage 1's witness JSONL, builds a per-row prompt by interpolating
the hint into the configured template, calls the teacher vLLM endpoint,
and writes a JSONL of (row_id, completion, boxed, accepted).

A row is ``accepted`` iff ``reject_if_boxed_mismatch`` is true and the
boxed answer in the completion equals ``kaggle_answer``. Otherwise the
completion is kept (caller can choose to filter later).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from tqdm import tqdm


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_boxed(text: str) -> str | None:
    """Return the contents of the LAST ``\\boxed{...}`` in ``text``.

    Strips surrounding whitespace inside the braces (the prompt template
    writes ``\\boxed{ X }`` to avoid YAML escaping issues; teachers may
    or may not preserve those spaces).
    """
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _render_prompt(template: str, row: dict[str, Any]) -> str:
    """Render the prompt template. Available placeholders:

    * ``{prompt}``                — the original Kaggle prompt
    * ``{kaggle_answer}``          — the stored answer
    * ``{rule_summary}``           — hint.rule_summary
    * ``{per_component}``          — hint.per_component (joined with newlines)
    * ``{extras.<key>}``           — anything in hint.extras (rendered inline)
    """
    hint = row.get("hint") or {}
    extras = hint.get("extras") or {}
    per_component = hint.get("per_component") or []
    per_component_str = "\n".join(
        f"  bit {i}: {rule}" if len(per_component) == 8 else f"  {rule}"
        for i, rule in enumerate(per_component)
    )
    ctx = {
        "prompt": row["prompt"],
        "kaggle_answer": row["kaggle_answer"],
        "rule_summary": hint.get("rule_summary", ""),
        "per_component": per_component_str,
        **{f"extras.{k}": v for k, v in extras.items() if isinstance(v, (str, int, float))},
    }
    # str.format would explode on { in YAML literals; do conservative replace
    out = template
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _call_vllm(base_url: str, model: str, prompt: str,
               sampling: dict[str, Any], timeout: float,
               assistant_prefill: str | None = None) -> str:
    """Call vLLM's chat/completions endpoint.

    Using chat (not completions) so the model's chat template handles
    role boundaries — otherwise vLLM's raw-completion mode treats our
    prompt's tail as part of the model's response stream and the output
    starts mid-sentence.

    If ``assistant_prefill`` is given, we send a 2-message conversation
    (user + assistant) and ask vLLM to *continue* the assistant message
    rather than start a new one. This is the prefilled-CoT path: the
    hint is embedded in the assistant prefill, the model continues from
    there, and the caller receives only the continuation (the prefill is
    re-prepended at write time so the stored CoT is one coherent string).
    """
    if assistant_prefill is None:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prefill},
            ],
            # vLLM-specific flags: keep the assistant message open and
            # don't tack on a new generation prompt.
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
    body["temperature"] = sampling.get("temperature", 0.3)
    body["max_tokens"]  = sampling.get("max_tokens", 4096)
    body["top_p"]       = sampling.get("top_p", 0.95)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def run(stage_cfg: dict[str, Any], stage1_out: Path,
        prompt_templates: dict[str, str], log) -> dict[str, int]:
    if not stage_cfg.get("enabled", True):
        log("  stage_2_teacher_distill: disabled")
        return {"skipped": True}

    endpoint = stage_cfg["endpoint"]
    base_url = endpoint["base_url"]
    model    = endpoint["model"]
    sampling = stage_cfg.get("sampling", {})
    # pass@k: draw k samples per row, keep accepted samples per `keep_policy`:
    #   "all"           → write every accepted sample as its own training row
    #                     (most data, may include duplicates when teacher is sharp)
    #   "first_correct" → stop after the first accepted sample per row, classic pass@k
    #                     (de-duplicates on row, doesn't waste budget after a hit)
    # k defaults to sampling.n_samples_per_row for back-compat.
    k = int(stage_cfg.get("k",
            sampling.get("n_samples_per_row", 1)))
    keep_policy = stage_cfg.get("keep_policy", "all")
    if keep_policy not in ("all", "first_correct"):
        raise ValueError(f"unknown keep_policy {keep_policy!r}")
    parallelism = int(stage_cfg.get("parallelism", 4))
    template_key = stage_cfg["prompt_template_key"]
    template = prompt_templates[template_key]
    prefill_key = stage_cfg.get("prefill_template_key")
    prefill_template = prompt_templates[prefill_key] if prefill_key else None
    reject_mismatch = bool(stage_cfg.get("reject_if_boxed_mismatch", True))
    expected_pass_rate = float(stage_cfg.get("expected_pass_rate", 0.0))
    timeout = float(stage_cfg.get("timeout_seconds", 180.0))
    out_path = Path(stage_cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in stage1_out.read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("status") == "ok"]
    log(f"  stage_2: {len(rows)} rows × pass@k={k} keep={keep_policy}"
        f"  prefill={'on' if prefill_template else 'off'}")

    counts = {"calls": 0, "accepted": 0, "boxed_mismatch": 0,
              "no_boxed": 0, "http_error": 0, "kept": 0}

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
        # Stored CoT = prefill + model's continuation.
        completion = (prefill + continuation) if prefill else continuation
        boxed = _extract_boxed(completion)
        bucket = "accepted"
        if not boxed:
            bucket = "no_boxed"
        elif reject_mismatch and boxed != row["kaggle_answer"]:
            bucket = "boxed_mismatch"
        return {
            "row_id": row["row_id"],
            "sample_k": j,
            "prompt": row["prompt"],
            "kaggle_answer": row["kaggle_answer"],
            "completion": completion,
            "boxed": boxed,
            "accepted": (bucket == "accepted"),
            "_bucket": bucket,
        }

    def _process_row(row: dict[str, Any]) -> list[dict[str, Any]]:
        """Run pass@k on one row; honor keep_policy.

        ``all``:           fire k calls, return every result (caller filters).
        ``first_correct``: fire calls sequentially j=0..k-1, stop early on
                           first accepted result. Returns 1..k results — the
                           tail are failures preceding the hit.
        """
        results: list[dict[str, Any]] = []
        for j in range(k):
            res = _one_call(row, j)
            results.append(res)
            if keep_policy == "first_correct" and res.get("accepted"):
                break
        return results

    # Stream results as futures complete so progress is visible live.
    # For "all": fan out all rows×k calls at once → maximum throughput,
    #   every accepted sample becomes a training row.
    # For "first_correct": one future per row that runs j=0..k-1
    #   sequentially and stops on the first accepted hit → saves budget,
    #   only one training row per Kaggle id.
    if keep_policy == "all":
        total_calls_planned = len(rows) * k
        unit_label = "calls"
    else:
        total_calls_planned = len(rows)
        unit_label = "rows"

    def _consume_one(res, out) -> None:
        counts["calls"] += 1
        bucket = res.pop("_bucket", "accepted")
        counts[bucket] = counts.get(bucket, 0) + 1
        if keep_policy == "first_correct" and not res.get("accepted"):
            return
        out.write(json.dumps(res) + "\n")
        out.flush()
        counts["kept"] += 1

    with out_path.open("w") as out, cf.ThreadPoolExecutor(parallelism) as ex:
        if keep_policy == "all":
            futures = [ex.submit(_one_call, row, j)
                       for row in rows for j in range(k)]
        else:  # first_correct
            futures = [ex.submit(_process_row, row) for row in rows]

        bar = tqdm(
            total=total_calls_planned, unit=unit_label, smoothing=0.05,
            mininterval=2.0, maxinterval=30.0,
            desc="stage_2", file=sys.stderr,
            dynamic_ncols=True,
        )
        log_every_n = max(1, total_calls_planned // 100)
        for fut in cf.as_completed(futures):
            result_bag = fut.result()
            if not isinstance(result_bag, list):
                result_bag = [result_bag]
            for res in result_bag:
                _consume_one(res, out)
                if counts["calls"] % log_every_n == 0:
                    log(f"    progress {counts['calls']}/{total_calls_planned} "
                        f"accepted={counts.get('accepted',0)} "
                        f"mismatch={counts.get('boxed_mismatch',0)} "
                        f"kept={counts['kept']}")
            bar.update(1)
            bar.set_postfix(
                accepted=counts["accepted"],
                kept=counts["kept"],
                refresh=False,
            )
        bar.close()

    # pass@k row_pass_rate = fraction of Kaggle rows where ≥1 of k samples
    # was accepted. This is what reject_if_boxed_mismatch is gating on.
    rows_with_hit = len({r["row_id"] for r in
                         (json.loads(l) for l in
                          out_path.read_text().splitlines() if l.strip())
                         if r.get("accepted")})
    row_pass_rate = rows_with_hit / max(1, len(rows))
    pass_rate = counts["accepted"] / max(1, counts["calls"])
    log(f"  stage_2: rows_with_hit={rows_with_hit}/{len(rows)} "
        f"(pass@{k}={row_pass_rate:.3f})  "
        f"per_call_accepted={counts['accepted']}/{counts['calls']} "
        f"({pass_rate:.3f})  mismatch={counts['boxed_mismatch']} "
        f"no_boxed={counts['no_boxed']} http_err={counts['http_error']} "
        f"kept={counts['kept']}")
    log(f"  wrote {out_path}")
    if row_pass_rate < expected_pass_rate:
        raise RuntimeError(
            f"stage_2 pass@{k} {row_pass_rate:.3f} < threshold "
            f"{expected_pass_rate:.3f}; halt to fix prompt"
        )
    return counts
