"""Eval worker — executes an :class:`EvalPlan`.

Two paths:

* **smoke** — writes a deterministic ``metrics.json`` + ``predictions.jsonl``
  derived from ``eval/local_holdout_small.jsonl``. Used by the PR8 seed
  workspace and all unit tests.
* **non-smoke** — loads the base model + LoRA adapter via vLLM and scores the
  dev split using a pluggable benchmark scorer. This is what the
  ``kaggle_nemo_reasoner`` workspace invokes for real H200 runs.

The non-smoke path is tightly coupled to vLLM + the Kaggle host contract (chat
template, max_model_len 4096, max_lora_rank 32, temperature 1.0, top_p 1.0,
max_tokens 3584). Matching those knobs verbatim is what lets the dev score
correlate with the leaderboard.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ..types import EvalPlan

logger = logging.getLogger(__name__)


def run_eval_plan(
    plan: EvalPlan,
    *,
    smoke: bool = True,
    benchmark: Any | None = None,
    workspace: Any | None = None,
    split: str | None = None,
) -> Path:
    out = Path(plan.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if smoke:
        _write_smoke_artifacts(plan, out)
        return out

    return _run_vllm_eval(plan, out, benchmark=benchmark, workspace=workspace, split=split)


# ── Smoke path (no GPU) ──────────────────────────────────────────────────

def _write_smoke_artifacts(plan: EvalPlan, out: Path) -> None:
    rows = _load_holdout_rows(plan)
    if rows:
        correct = sum(1 for row in rows if row.get("is_correct"))
        primary = correct / max(1, len(rows))
    else:
        primary = 0.0
    metrics = {
        "local_holdout_pass_at_1": primary,
        "primary": primary,
        "format_error_rate": 0.0,
        "avg_output_tokens": 0.0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "predictions.jsonl", "w") as f:
        for row in rows or []:
            f.write(json.dumps(row) + "\n")


def _load_holdout_rows(plan: EvalPlan) -> list[dict]:
    root = Path(plan.config_path).parent
    candidates = [root / "local_holdout_small.jsonl", root / "local_holdout.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            rows: list[dict] = []
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return rows
    return []


# ── Non-smoke path (vLLM + LoRA) ─────────────────────────────────────────

def _run_vllm_eval(
    plan: EvalPlan,
    out: Path,
    *,
    benchmark: Any | None,
    workspace: Any | None,
    split: str | None,
) -> Path:
    """Load base model + LoRA via vLLM and score the dev split.

    The caller must pass a ``benchmark`` that exposes ``load_dev_rows`` and
    ``score_predictions`` (see
    :class:`agent_evolve.benchmarks.nemo_reason.kaggle.KaggleNemoReasonerBenchmark`).
    """
    if benchmark is None or workspace is None:
        raise RuntimeError(
            "Non-smoke eval requires `benchmark` and `workspace` to be passed through."
        )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = dict(plan.generation_config or {})
    model_path = cfg.get("model_path")
    if not model_path:
        raise RuntimeError(
            "plan.generation_config['model_path'] is required for non-smoke eval."
        )
    limit = cfg.get("limit")
    tp = int(cfg.get("tensor_parallel_size", 1))
    seed = int(cfg.get("seed", 0))
    max_model_len = int(cfg.get("max_model_len", 4096))
    max_lora_rank = int(cfg.get("max_lora_rank", 32))
    gpu_memory_utilization = float(cfg.get("gpu_memory_utilization", 0.85))

    adapter_path = plan.checkpoint.path
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"adapter not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    engine_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp,
        seed=seed,
        max_model_len=max_model_len,
        max_lora_rank=max_lora_rank,
        max_num_seqs=int(cfg.get("max_num_seqs", 128)),
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="auto",
        trust_remote_code=True,
    )
    logger.info("[eval] loading vLLM engine: %s", {k: v for k, v in engine_kwargs.items() if k != "model"})
    llm = LLM(**engine_kwargs)

    sampling = SamplingParams(
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        max_tokens=int(cfg.get("max_tokens", 3584)),
        seed=seed,
    )

    split_name = split or plan.split
    dev_rows = benchmark.load_dev_rows(workspace, split_name)
    if limit is not None:
        dev_rows = dev_rows[: int(limit)]
    logger.info("[eval] %d dev rows on split=%s", len(dev_rows), split_name)

    prompts = [_build_prompt(row.prompt, tokenizer) for row in dev_rows]

    lora_request = LoRARequest(plan.checkpoint.name or "candidate", 1, adapter_path)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_request)
    elapsed = time.time() - t0
    logger.info("[eval] generate done in %.1f min", elapsed / 60.0)

    raw_texts = [out.outputs[0].text for out in outputs]
    metrics, buckets, records = benchmark.score_predictions(dev_rows, raw_texts)

    metrics_json = {
        "overall_accuracy": metrics.primary_metric_value,
        metrics.primary_metric_name: metrics.primary_metric_value,
        "per_domain": _per_domain(metrics.secondary),
        "n_eval": int(metrics.secondary.get("n_eval", len(dev_rows))),
        "seed": seed,
        "adapter": adapter_path,
        "model_path": model_path,
        "eval_seconds": elapsed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics_json, indent=2))
    with open(out / "predictions.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(out / "raw_outputs.jsonl", "w") as f:
        for row, text in zip(dev_rows, raw_texts):
            f.write(json.dumps({"id": row.id, "text": text}) + "\n")
    (out / "error_buckets.json").write_text(
        json.dumps({"counts": buckets.counts}, indent=2)
    )
    return out


def _build_prompt(raw_prompt: str, tokenizer: Any) -> str:
    # Mirror the Kaggle host prompt: raw_prompt + boxed-instruction suffix,
    # wrapped in the model's chat template with add_generation_prompt=True.
    user_content = raw_prompt + (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception:
        return user_content


def _per_domain(secondary: dict[str, float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, value in secondary.items():
        if not key.startswith("domain."):
            continue
        _, domain, field = key.split(".", 2)
        out.setdefault(domain, {})[field] = value
    return out
