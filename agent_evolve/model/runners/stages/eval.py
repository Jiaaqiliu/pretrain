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

The Kaggle host contract (chat template, max_model_len 8192, max_lora_rank
32, temperature 0.0 greedy, top_p 1.0, max_tokens 7680, max_num_seqs 64)
is preserved on the LoRA path so the dev score correlates with the
leaderboard. These values match the Kaggle Evaluation page's runtime args,
not metric.score()'s python-level defaults — those two disagree.
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

    # Engine routing. Default is vLLM (fast, production path). The HF
    # backend exists as a fallback for hosts where vLLM's triton JIT
    # fails — e.g., the k8s nodes on driver 570.148 — but HF's generate()
    # path through Nemotron-3-Nano's cuda_kernels_forward has several
    # latent bugs (cache class threading, weight shape mismatches) so
    # only use it when vLLM is genuinely blocked.
    cfg = dict(plan.generation_config or {})
    engine = str(cfg.get("engine", "vllm")).lower()
    if engine == "vllm":
        return _run_vllm_eval(plan, out, benchmark=benchmark, workspace=workspace, split=split)
    if engine == "hf":
        return _run_hf_eval(plan, out, benchmark=benchmark, workspace=workspace, split=split)
    raise ValueError(f"unknown eval engine: {engine!r} (expected 'hf' or 'vllm')")


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
    # DP≥2 shards prompts across replicas. On this cluster TP=8 eval crashes
    # inside vLLM's multiproc TP-worker path with "device kernel image is
    # invalid" during profile_run's Inductor/Triton JIT. TP=1 DP=8 sidesteps
    # the shared TP codepath entirely — each engine is a single-GPU process
    # loading its own weights. Slower model load (8 copies × ~4 min) but
    # 8× more generation throughput at eval time, so net wins for any eval
    # over ~50 prompts. Fit: Nemotron-3-Nano BF16 is ~18 GB; H200 has 144 GB.
    dp = int(cfg.get("data_parallel_size", 1))
    seed = int(cfg.get("seed", 0))
    max_model_len = int(cfg.get("max_model_len", 8192))
    gpu_memory_utilization = float(cfg.get("gpu_memory_utilization", 0.85))

    # Default ON for eval: skips torch.compile + Inductor autotune. Required on
    # k8s nodes whose NVIDIA driver (570-series) predates torch 2.10's cu128
    # kernel targets — observed failure is "CUDA driver error: device kernel
    # image is invalid" during Inductor's generate_and_run_autotune_block at
    # vLLM profile_run. Perf cost on eval is negligible; correctness wins.
    # Override via cfg["enforce_eager"]=false when running on a node whose
    # driver matches the image (e.g. a local smoke test).
    enforce_eager = bool(cfg.get("enforce_eager", True))

    if is_full_model:
        # Full-state checkpoint: vLLM loads it directly as the model.
        # Tokenizer comes from the same dir (saved by FullDeepspeedAdapter).
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        engine_kwargs = dict(
            model=checkpoint_path,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            seed=seed,
            max_model_len=max_model_len,
            max_num_seqs=int(cfg.get("max_num_seqs", 64)),
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="auto",
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )
    else:
        # LoRA adapter: vLLM loads the base model with LoRA support enabled.
        max_lora_rank = int(cfg.get("max_lora_rank", 32))
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        engine_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            seed=seed,
            max_model_len=max_model_len,
            max_lora_rank=max_lora_rank,
            max_num_seqs=int(cfg.get("max_num_seqs", 64)),
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            dtype="auto",
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )

    logger.info(
        "[eval] loading vLLM engine (full_model=%s): %s",
        is_full_model,
        {k: v for k, v in engine_kwargs.items() if k != "model"},
    )
    llm = LLM(**engine_kwargs)

    sampling_kwargs: dict[str, Any] = dict(
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        max_tokens=int(cfg.get("max_tokens", 7680)),
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


def _run_hf_eval(
    plan: EvalPlan,
    out: Path,
    *,
    benchmark: Any | None,
    workspace: Any | None,
    split: str | None,
) -> Path:
    """HF transformers backend — no vLLM, no triton JIT.

    Writes the same artifact set as the vLLM path (metrics.json,
    predictions.jsonl, raw_outputs.jsonl, error_buckets.json) so callers
    downstream are indifferent to which engine produced them.

    Adapter loading is done via PEFT. Full-state checkpoints load the
    directory as a standalone model with ``from_pretrained``.
    """
    if benchmark is None or workspace is None:
        raise RuntimeError(
            "Non-smoke eval requires `benchmark` and `workspace` to be passed through."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    seed = int(cfg.get("seed", 0))
    max_new_tokens = int(cfg.get("max_tokens", 7680))
    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 1.0))
    batch_size = int(cfg.get("hf_batch_size", 4))

    torch.manual_seed(seed)

    if is_full_model:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",           # shards across visible GPUs
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # PEFT plugs the adapter onto the base. Works for LoRA / DoRA / IA³.
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, checkpoint_path)

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"    # generation needs left-padding

    split_name = split or plan.split
    dev_rows = benchmark.load_dev_rows(workspace, split_name)
    if limit is not None:
        dev_rows = dev_rows[: int(limit)]
    logger.info("[eval:hf] %d dev rows on split=%s", len(dev_rows), split_name)

    prompts = [_resolve_prompt(benchmark, row, tokenizer) for row in dev_rows]

    # Sampling vs greedy: HF `generate` interprets do_sample=False as greedy.
    # Kaggle contract uses temperature=0 which we map to greedy here.
    do_sample = temperature > 0.0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    # Nemotron-H is a hybrid (Mamba + attention) model, but its modeling
    # file ships with several bugs that break generate() throughput:
    #
    # Bug 1: forward() takes `cache_params`, but HF's generate() threads
    #   `past_key_values`. Without a patch, every new token recomputes
    #   the full context → O(T²) generation cost.
    # Bug 2: the cache __init__ doesn't store `conv_kernel_size`, but
    #   cuda_kernels_forward references `cache_params.conv_kernel_size`.
    # Bug 3: update_conv_state / update_ssm_state do
    #   `self.conv_states.device` where conv_states is a Python list, not
    #   a tensor. Same for `.zero_()` in reset().
    #
    # All three patched here, idempotent.
    base_for_patch = model.base_model.model if hasattr(model, "base_model") else model
    if not getattr(base_for_patch, "_cache_param_patched", False):
        _raw_forward = base_for_patch.forward
        cfg_obj = base_for_patch.config
        conv_kernel_size = int(getattr(cfg_obj, "conv_kernel", 4))

        def _forward_with_cache(*args, **kwargs):
            pkv = kwargs.pop("past_key_values", None)
            if pkv is not None and kwargs.get("cache_params") is None:
                if not hasattr(pkv, "conv_kernel_size"):
                    pkv.conv_kernel_size = conv_kernel_size
                # Patch the instance's bad methods. The model file's
                # update_conv_state/update_ssm_state call `.device` on
                # lists. Rewrite to get device from the slot tensor.
                if not getattr(pkv, "_update_methods_patched", False):
                    def _upd_conv(layer_idx, new_conv_state, cache_init=False):
                        dev = pkv.conv_states[layer_idx].device
                        if cache_init:
                            pkv.conv_states[layer_idx] = new_conv_state.to(dev)
                        else:
                            pkv.conv_states[layer_idx] = (
                                pkv.conv_states[layer_idx].roll(shifts=-1, dims=-1)
                            )
                            pkv.conv_states[layer_idx][:, :, -1] = (
                                new_conv_state[:, 0, :].to(dev)
                            )
                        return pkv.conv_states[layer_idx]
                    def _upd_ssm(layer_idx, new_ssm_state):
                        dev = pkv.ssm_states[layer_idx].device
                        pkv.ssm_states[layer_idx] = new_ssm_state.to(dev)
                        return pkv.ssm_states[layer_idx]
                    pkv.update_conv_state = _upd_conv
                    pkv.update_ssm_state = _upd_ssm
                    pkv._update_methods_patched = True
                kwargs["cache_params"] = pkv
            return _raw_forward(*args, **kwargs)

        base_for_patch.forward = _forward_with_cache  # type: ignore[method-assign]
        base_for_patch._cache_param_patched = True

    raw_texts: list[str] = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(cfg.get("max_model_len", 8192)) - max_new_tokens,
        ).to(model.device)
        with torch.inference_mode():
            gen_out = model.generate(**enc, **gen_kwargs)
        # Strip the prompt prefix per row — HF returns full [prompt+completion].
        prompt_lens = enc["input_ids"].shape[1]
        completions = gen_out[:, prompt_lens:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        raw_texts.extend(decoded)
        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(
                "[eval:hf] %d/%d rows  elapsed=%.1fm  rate=%.1f rows/s",
                i + len(batch_prompts), len(prompts),
                elapsed / 60.0,
                (i + len(batch_prompts)) / max(elapsed, 1e-6),
            )
    elapsed = time.time() - t0
    logger.info("[eval:hf] generate done in %.1f min (%d rows)",
                elapsed / 60.0, len(prompts))

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
        "engine": "hf",
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
