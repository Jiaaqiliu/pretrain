"""Eval worker — executes an :class:`EvalPlan`.

Two paths:

* **smoke** — writes a deterministic ``metrics.json`` + ``predictions.jsonl``
  derived from ``eval/local_holdout_small.jsonl``. Used by the PR8 seed
  workspace and all unit tests.
* **non-smoke** — loads the model via vLLM and scores the dev split using a
  pluggable benchmark scorer. This is what the ``kaggle_nemo_reasoner``
  workspace invokes for real H200 runs.

Checkpoint dispatch (non-smoke): :attr:`CheckpointRef.kind` determines how
vLLM loads the candidate.

* ``"adapter"`` — load base model + LoRA adapter via ``LoRARequest`` (the
  default; what every LoRA / DoRA / IA³ ``ModelAdapter`` produces).
* ``"full_state"`` — load ``checkpoint.path`` directly as the model; no
  ``LoRARequest``. Produced by full-parameter adapters
  (:class:`FullDeepspeedAdapter`).

The Kaggle host contract (chat template, max_model_len 4096, max_lora_rank
32, temperature 1.0, top_p 1.0, max_tokens 3584) is preserved on the LoRA
path so the dev score still correlates with the leaderboard.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ...types import EvalPlan

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
    """Load the candidate via vLLM and score the dev split.

    Branches on ``plan.checkpoint.kind``:

    * ``"full_state"`` — load ``checkpoint.path`` as the model. Tokenizer is
      loaded from the same dir (the full-param adapter saved it there).
    * Anything else (default ``"adapter"``) — load the base model
      (``plan.generation_config['model_path']``) and apply the LoRA
      ``LoRARequest`` per generate call.

    The caller must pass a ``benchmark`` that exposes ``load_dev_rows`` and
    ``score_predictions``. Optional benchmark hooks: ``build_eval_prompt``
    overrides the legacy Kaggle / boxed-answer suffix per
    :func:`benchmarks.helpers.build_eval_prompt`.
    """
    if benchmark is None or workspace is None:
        raise RuntimeError(
            "Non-smoke eval requires `benchmark` and `workspace` to be passed through."
        )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from ....benchmarks.helpers import build_eval_prompt as _resolve_prompt

    cfg = dict(plan.generation_config or {})
    model_path = cfg.get("model_path")
    if not model_path:
        raise RuntimeError(
            "plan.generation_config['model_path'] is required for non-smoke eval."
        )

    checkpoint_path = plan.checkpoint.path
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    is_full_model = plan.checkpoint.kind == "full_state"

    limit = cfg.get("limit")
    tp = int(cfg.get("tensor_parallel_size", 1))
    seed = int(cfg.get("seed", 0))
    max_model_len = int(cfg.get("max_model_len", 4096))
    gpu_memory_utilization = float(cfg.get("gpu_memory_utilization", 0.85))

    if is_full_model:
        # Full-state checkpoint: vLLM loads it directly as the model.
        # Tokenizer comes from the same dir (saved by FullDeepspeedAdapter).
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        engine_kwargs = dict(
            model=checkpoint_path,
            tensor_parallel_size=tp,
            seed=seed,
            max_model_len=max_model_len,
            max_num_seqs=int(cfg.get("max_num_seqs", 128)),
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="auto",
            trust_remote_code=True,
        )
    else:
        # LoRA adapter: vLLM loads the base model with LoRA support enabled.
        max_lora_rank = int(cfg.get("max_lora_rank", 32))
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

    logger.info(
        "[eval] loading vLLM engine (full_model=%s): %s",
        is_full_model,
        {k: v for k, v in engine_kwargs.items() if k != "model"},
    )
    llm = LLM(**engine_kwargs)

    sampling_kwargs: dict[str, Any] = dict(
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        max_tokens=int(cfg.get("max_tokens", 3584)),
        seed=seed,
    )
    rep_penalty = cfg.get("repetition_penalty")
    if rep_penalty is not None:
        sampling_kwargs["repetition_penalty"] = float(rep_penalty)
    sampling = SamplingParams(**sampling_kwargs)

    split_name = split or plan.split
    dev_rows = benchmark.load_dev_rows(workspace, split_name)
    if limit is not None:
        dev_rows = dev_rows[: int(limit)]
    logger.info("[eval] %d dev rows on split=%s", len(dev_rows), split_name)

    # Prefer ``benchmark.build_eval_prompt(row, tokenizer)`` if implemented;
    # falls back to the legacy nemo_reasoner Kaggle suffix builder.
    prompts = [_resolve_prompt(benchmark, row, tokenizer) for row in dev_rows]

    generate_kwargs: dict[str, Any] = {"sampling_params": sampling}
    if not is_full_model:
        from vllm.lora.request import LoRARequest
        from ....backends.tinkerlite.adapters import resolve_adapter

        # Use the registered adapter's request builder when adapter.yaml::type
        # is known (LoRA today; future DoRA / IA³ may want different
        # request shapes). Falls back to plain LoRARequest for legacy
        # workspaces that don't declare a type.
        adapter_kind = _read_adapter_kind(workspace)
        lora_request = None
        if adapter_kind:
            try:
                lora_request = resolve_adapter(adapter_kind).vllm_lora_request(plan.checkpoint)
            except KeyError:
                lora_request = None
        if lora_request is None:
            lora_request = LoRARequest(plan.checkpoint.name or "candidate", 1, checkpoint_path)
        generate_kwargs["lora_request"] = lora_request

    t0 = time.time()
    outputs = llm.generate(prompts, **generate_kwargs)
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
        "checkpoint": checkpoint_path,
        "checkpoint_kind": plan.checkpoint.kind,
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


def _read_adapter_kind(workspace: Any) -> str | None:
    """Read ``model/adapter.yaml::type`` if present; ``None`` otherwise.

    Pure helper — kept here (not in helpers.dataset) to avoid importing
    yaml in the hot path of LoRA-only callers; this only fires once per
    eval invocation.
    """
    import yaml as _yaml

    path = Path(workspace.root) / "model" / "adapter.yaml"
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            data = _yaml.safe_load(f) or {}
    except Exception:  # pragma: no cover — best-effort config load
        return None
    kind = data.get("type")
    return str(kind) if kind else None


def _per_domain(secondary: dict[str, float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, value in secondary.items():
        if not key.startswith("domain."):
            continue
        _, domain, field = key.split(".", 2)
        out.setdefault(domain, {})[field] = value
    return out
