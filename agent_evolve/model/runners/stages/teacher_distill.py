"""Teacher-distillation synthesis worker.

Pipeline stage ``synth_generate`` sends prompts from ``data/splits/train_local.csv``
(the same split-table used by the Kaggle training set) to a teacher model,
filters rollouts by correctness + length, and emits a JSONL file that
subsequent SFT stages consume via ``data/sources.yaml``.

Two teacher providers are supported; pick one in ``teacher_distill.yaml``
via ``teacher_provider:``:

  * ``vllm_local`` — load a local model via vLLM (default Nemotron-3-Super-
    120B-FP8, TP=8). Heavy: loads 117 GiB of weights on 8 H200s, runs in
    a subprocess so CUDA state is reclaimed before SFT.
  * ``bedrock`` — call AWS Bedrock Converse against a managed model
    (default ``us.anthropic.claude-sonnet-4-6-v1:0``). No GPU, no
    subprocess. Uses boto3 + the host's ambient AWS creds.

Filter mirrors the verified recipe at ``../nemotron-auto-research/scripts/
gen_teacher_traces.py``:
  * ``verify(answer, extract_final_answer(text))`` must be True
  * output must contain ``\\boxed{...}``
  * teacher_n_tokens >= ``min_tokens`` (protects against E-08 style-collapse)
  * student_n_tokens <= ``max_tokens`` (so SFT doesn't truncate targets)

The vllm_local teacher leaves residual CUDA state on every worker GPU
that PyTorch does not release on ``del llm + empty_cache()``. To keep
the subsequent SFT stage from OOMing on GPU 0, vllm_local runs in a
SUBPROCESS by default (via ``python -m agent_evolve.model.runners.stages.
teacher_distill``), so the OS reclaims the teacher's CUDA state when the
subprocess exits. Set ``AE_SYNTH_SUBPROCESS=0`` to force in-process
(e.g. for standalone debugging). The bedrock provider is always
in-process — no subprocess needed since there's no GPU state to clean up.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ....benchmarks.nemo_reasoner import (
    EVAL_INSTRUCTION_SUFFIX,
    build_eval_prompt,
    extract_final_answer,
    verify,
)


# Flashinfer compile errors on this cluster; disable fast paths.
# Mirrors CLAUDE.md §E-38 documented workaround.
_FLASHINFER_OFF = {
    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0",
    "VLLM_USE_FLASHINFER_MOE_FP8": "0",
    "VLLM_USE_FLASHINFER_MOE_FP4": "0",
    "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
}


# ── Public entry point ───────────────────────────────────────────────────

def run_synth_stage(
    workspace: Any,
    stage: dict,
    *,
    smoke: bool = True,
    budget_seconds: float | None = None,  # noqa: ARG001 — teacher gen owns its own clock
) -> tuple[Path, dict[str, Any]]:
    """Run one teacher-distillation stage. Returns (jsonl_path, stats_dict)."""
    if smoke:
        return _run_smoke_synth(workspace, stage)
    return _run_real_synth(workspace, stage)


# ── Smoke path (no GPU) ──────────────────────────────────────────────────

def _run_smoke_synth(workspace: Any, stage: dict) -> tuple[Path, dict[str, Any]]:
    outdir = Path(workspace.root) / "data" / "synth"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{stage.get('name', 'teacher_distill')}.jsonl"
    rows = [
        {
            "id": f"smoke-{i}",
            "domain": "bits",
            "answer": "00000000",
            "prompt_rendered": "smoke prompt\n",
            "completion": "Let me think.\\boxed{00000000}<|im_end|>",
            "source_experiment": "smoke",
            "teacher_n_tokens": 10,
            "student_n_tokens": 10,
        }
        for i in range(4)
    ]
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = {
        "kept": len(rows),
        "total": len(rows),
        "dropped": {},
        "per_domain_kept": {"bits": len(rows)},
    }
    _append_to_sources(workspace, out_path)
    return out_path, stats


# ── Real path (subprocess-isolated by default) ───────────────────────────

def _run_real_synth(workspace: Any, stage: dict) -> tuple[Path, dict[str, Any]]:
    cfg = _load_real_synth_config(workspace, stage)
    provider = cfg.get("teacher_provider", "vllm_local")
    # The bedrock provider has no CUDA state to reclaim; subprocess
    # isolation only matters for the vLLM path.
    use_subprocess = (
        provider == "vllm_local"
        and os.environ.get("AE_SYNTH_SUBPROCESS", "1") != "0"
    )
    if use_subprocess:
        _run_real_synth_subprocess(cfg)
    else:
        _run_real_synth_inproc(cfg)

    out_path = Path(cfg["out_path"])
    stats_path = out_path.with_suffix(".stats.json")
    if not stats_path.exists():
        raise RuntimeError(
            f"synth stage did not produce {stats_path} — check teacher-subprocess log"
        )
    stats = json.loads(stats_path.read_text())

    # Optional verifier_gate: defer actual filtering to a follow-up because
    # teacher-output JSONL has its own row schema (prompt_rendered /
    # completion) distinct from GeneratedRow. For now we just record that
    # the gate was requested so pipelines / stats consumers know the
    # intent and can detect misconfiguration (gate on but no verifiers
    # registered on the benchmark).
    if stage.get("verifier_gate"):
        stats["verifier_gate_requested"] = True

    _append_to_sources(workspace, out_path)
    return out_path, stats


def _run_real_synth_subprocess(cfg: dict) -> None:
    """Launch the real synth in a child Python process.

    When the child exits, the OS releases every byte of CUDA state the
    teacher's TP workers were holding. The parent then sees a clean GPU for
    the subsequent SFT stage.
    """
    cfg_path = Path(cfg["out_path"]).with_suffix(".cfg.json")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    env = os.environ.copy()
    for k, v in _FLASHINFER_OFF.items():
        env.setdefault(k, v)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # If the driver told us which GPUs the teacher should use, expose them
    # to the child. This lets the parent keep only one GPU visible for SFT
    # while the teacher subprocess still gets its TP=4 quota.
    synth_gpus = os.environ.get("AE_SYNTH_GPUS")
    if synth_gpus:
        env["CUDA_VISIBLE_DEVICES"] = synth_gpus

    cmd = [
        sys.executable,
        "-m",
        "agent_evolve.model.runners.stages.teacher_distill",
        "--config",
        str(cfg_path),
    ]
    print(f"[synth] spawning teacher subprocess: {' '.join(cmd)}")
    # Stream stdout/stderr through to the parent's log via the default
    # inheritance. ``check=True`` raises on non-zero exit so the backend
    # can flip the trial status to ``train_failed``.
    subprocess.run(cmd, env=env, check=True)


def _run_real_synth_inproc(cfg: dict) -> None:
    """Dispatch to the right provider. Called directly or as __main__."""
    provider = cfg.get("teacher_provider", "vllm_local")
    if provider == "vllm_local":
        _run_real_synth_inproc_vllm(cfg)
    elif provider == "bedrock":
        _run_real_synth_inproc_bedrock(cfg)
    else:
        raise ValueError(
            f"unknown teacher_provider {provider!r}; "
            f"expected 'vllm_local' or 'bedrock'"
        )


def _sample_prompts(cfg: dict):
    """Shared prompt-sampling for both providers. Returns a DataFrame."""
    import pandas as pd
    df = pd.read_csv(cfg["prompts_csv"])
    sampled = []
    for domain in cfg["domains"]:
        g = df[df["domain"] == domain]
        k = min(cfg["per_domain"], len(g))
        if k == 0:
            continue
        sampled.append(g.sample(n=k, random_state=cfg["seed"]))
    return pd.concat(sampled).reset_index(drop=True)


def _run_real_synth_inproc_vllm(cfg: dict) -> None:
    """The heavy-lifting path. Loads the 120B and generates. Meant to be run
    either directly (AE_SYNTH_SUBPROCESS=0) or as ``__main__`` in a subprocess.
    """
    for k, v in _FLASHINFER_OFF.items():
        os.environ.setdefault(k, v)

    import pandas as pd  # noqa: F401 — used by _sample_prompts via closure
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_path = Path(cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_df = _sample_prompts(cfg)
    print(f"[synth] {len(prompts_df)} prompts across domains={cfg['domains']}")

    print(f"[synth] loading student tokenizer from {cfg['student_model_path']}")
    base_tok = AutoTokenizer.from_pretrained(
        cfg["student_model_path"], trust_remote_code=True
    )

    print(f"[synth] loading teacher {cfg['teacher_model_path']} (TP={cfg['tp']})")
    # enforce_eager skips torch.compile + inductor autotune. Required on nodes
    # whose NVIDIA driver predates torch 2.10's cu128 kernel targets (observed
    # on EKS driver 570 — triggers "device kernel image is invalid" during
    # FP8 MoE autotuning). The perf cost is ~10% at rollout time on BF16 but
    # matters less for FP8 MoE where kernel selection is already specialized.
    llm = LLM(
        model=cfg["teacher_model_path"],
        tensor_parallel_size=cfg["tp"],
        max_model_len=cfg["max_tokens"] + 1024,
        max_num_seqs=cfg["max_num_seqs"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        dtype="auto",
        trust_remote_code=True,
        enable_prefix_caching=False,  # prompts are unique
        enforce_eager=bool(cfg.get("enforce_eager", False)),
    )
    teacher_tok = llm.get_tokenizer()

    teacher_rendered: list[str] = []
    base_rendered: list[str] = []
    for row in prompts_df.itertuples(index=False):
        instr = row.prompt + EVAL_INSTRUCTION_SUFFIX
        try:
            tp = teacher_tok.apply_chat_template(
                [{"role": "user", "content": instr}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            tp = instr
        teacher_rendered.append(tp)
        base_rendered.append(build_eval_prompt(row.prompt, base_tok))

    sp = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_tokens"],
        n=cfg["n_samples"],
        seed=cfg["seed"],
    )

    t0 = time.time()
    outs = llm.generate(teacher_rendered, sampling_params=sp)
    wall = time.time() - t0
    print(f"[synth] generate done in {wall/60:.1f} min")

    kept = 0
    reasons: Counter[str] = Counter()
    per_dom_kept: Counter[str] = Counter()
    per_dom_attempt: Counter[str] = Counter()

    raw_path = out_path.with_suffix(".all.jsonl")
    with open(out_path, "w") as f_kept, open(raw_path, "w") as f_all:
        for i, (row, o, bp) in enumerate(
            zip(prompts_df.itertuples(index=False), outs, base_rendered)
        ):
            per_dom_attempt[row.domain] += 1
            for j, comp in enumerate(o.outputs):
                text = comp.text
                n_tok = len(comp.token_ids)
                pred = extract_final_answer(text)
                correct = bool(verify(str(row.answer), str(pred)))
                has_boxed = "\\boxed{" in text
                student_ids = base_tok.encode(text, add_special_tokens=False)
                over_student_max = len(student_ids) > cfg["max_tokens"]
                too_short = n_tok < cfg["min_tokens"]

                if not correct:
                    verdict = "wrong"
                elif not has_boxed:
                    verdict = "no_boxed"
                elif too_short:
                    verdict = "too_short"
                elif over_student_max:
                    verdict = "over_student_max"
                else:
                    verdict = "kept"
                reasons[verdict] += 1

                completion = text.rstrip() + "<|im_end|>"
                rec_raw = {
                    "id": f"teacher-{row.domain}-{i:05d}-{j}",
                    "domain": row.domain,
                    "answer": str(row.answer),
                    "prediction": pred,
                    "prompt_rendered": bp,
                    "completion": completion,
                    "source_experiment": cfg.get("stage_name", "teacher_distill"),
                    "teacher_n_tokens": n_tok,
                    "student_n_tokens": len(student_ids),
                    "correct": correct,
                    "has_boxed": has_boxed,
                    "verdict": verdict,
                }
                f_all.write(json.dumps(rec_raw) + "\n")
                if verdict == "kept":
                    f_kept.write(json.dumps({
                        k: rec_raw[k] for k in
                        ("id", "domain", "answer", "prompt_rendered",
                         "completion", "source_experiment",
                         "teacher_n_tokens", "student_n_tokens")
                    }) + "\n")
                    kept += 1
                    per_dom_kept[row.domain] += 1

    stats = {
        "kept": kept,
        "total": len(outs) * cfg["n_samples"],
        "dropped": dict(reasons),
        "per_domain_kept": dict(per_dom_kept),
        "per_domain_attempt": dict(per_dom_attempt),
        "wall_seconds": wall,
        "out_path": str(out_path),
        "raw_path": str(raw_path),
        "teacher_model_path": cfg["teacher_model_path"],
        "teacher_provider": "vllm_local",
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[synth] kept {kept} / {len(outs) * cfg['n_samples']} — {stats_path}")


def _run_real_synth_inproc_bedrock(cfg: dict) -> None:
    """Teacher = managed model on AWS Bedrock (e.g. Claude Sonnet 4.6).

    No GPU, no subprocess. Uses boto3 + ambient AWS creds. Output JSONL
    is byte-compatible with the vllm_local path so SFT consumes it
    unchanged.

    Notes on counting:
    - ``teacher_n_tokens`` is the Bedrock response's
      ``usage.outputTokens`` (Bedrock's own tokenizer, not the teacher's).
    - ``student_n_tokens`` uses the student base-model tokenizer for the
      len-cap check (same as the vllm_local path).
    """
    import time
    from transformers import AutoTokenizer

    out_path = Path(cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_df = _sample_prompts(cfg)
    print(f"[synth] {len(prompts_df)} prompts across domains={cfg['domains']} "
          f"(provider=bedrock model={cfg['teacher_model_path']})")

    print(f"[synth] loading student tokenizer from {cfg['student_model_path']}")
    base_tok = AutoTokenizer.from_pretrained(
        cfg["student_model_path"], trust_remote_code=True
    )

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "teacher_provider=bedrock requires boto3; install with "
            "`pip install boto3` in the driver venv"
        ) from exc

    region = cfg.get("bedrock_region") or os.environ.get(
        "AWS_REGION", "us-west-2"
    )
    client = boto3.client("bedrock-runtime", region_name=region)
    model_id = cfg["teacher_model_path"]

    kept = 0
    reasons: Counter[str] = Counter()
    per_dom_kept: Counter[str] = Counter()
    per_dom_attempt: Counter[str] = Counter()

    raw_path = out_path.with_suffix(".all.jsonl")
    base_rendered: list[str] = []
    for row in prompts_df.itertuples(index=False):
        base_rendered.append(build_eval_prompt(row.prompt, base_tok))

    t0 = time.time()
    with open(out_path, "w") as f_kept, open(raw_path, "w") as f_all:
        for i, (row, bp) in enumerate(
            zip(prompts_df.itertuples(index=False), base_rendered)
        ):
            per_dom_attempt[row.domain] += 1
            instr = row.prompt + EVAL_INSTRUCTION_SUFFIX
            for j in range(int(cfg["n_samples"])):
                text, n_tok, err = _bedrock_generate(
                    client,
                    model_id=model_id,
                    user_message=instr,
                    max_tokens=cfg["max_tokens"],
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                )
                if err is not None:
                    print(f"[synth] bedrock error on row {i} sample {j}: {err}")
                    reasons["bedrock_error"] += 1
                    continue

                pred = extract_final_answer(text)
                correct = bool(verify(str(row.answer), str(pred)))
                has_boxed = "\\boxed{" in text
                student_ids = base_tok.encode(text, add_special_tokens=False)
                over_student_max = len(student_ids) > cfg["max_tokens"]
                too_short = n_tok < cfg["min_tokens"]

                if not correct:
                    verdict = "wrong"
                elif not has_boxed:
                    verdict = "no_boxed"
                elif too_short:
                    verdict = "too_short"
                elif over_student_max:
                    verdict = "over_student_max"
                else:
                    verdict = "kept"
                reasons[verdict] += 1

                completion = text.rstrip() + "<|im_end|>"
                rec_raw = {
                    "id": f"teacher-{row.domain}-{i:05d}-{j}",
                    "domain": row.domain,
                    "answer": str(row.answer),
                    "prediction": pred,
                    "prompt_rendered": bp,
                    "completion": completion,
                    "source_experiment": cfg.get("stage_name", "teacher_distill"),
                    "teacher_n_tokens": n_tok,
                    "student_n_tokens": len(student_ids),
                    "correct": correct,
                    "has_boxed": has_boxed,
                    "verdict": verdict,
                }
                f_all.write(json.dumps(rec_raw) + "\n")
                if verdict == "kept":
                    f_kept.write(json.dumps({
                        k: rec_raw[k] for k in
                        ("id", "domain", "answer", "prompt_rendered",
                         "completion", "source_experiment",
                         "teacher_n_tokens", "student_n_tokens")
                    }) + "\n")
                    kept += 1
                    per_dom_kept[row.domain] += 1

    wall = time.time() - t0
    total = len(prompts_df) * int(cfg["n_samples"])
    stats = {
        "kept": kept,
        "total": total,
        "dropped": dict(reasons),
        "per_domain_kept": dict(per_dom_kept),
        "per_domain_attempt": dict(per_dom_attempt),
        "wall_seconds": wall,
        "out_path": str(out_path),
        "raw_path": str(raw_path),
        "teacher_model_path": model_id,
        "teacher_provider": "bedrock",
        "bedrock_region": region,
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[synth] kept {kept} / {total} — {stats_path}")


def _bedrock_generate(
    client,
    *,
    model_id: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int = 5,
) -> tuple[str | None, int, str | None]:
    """One Bedrock converse call. Returns (text, output_tokens, error_message_or_None).

    Newer Claude Sonnet models reject ``temperature`` and ``topP`` being
    sent together; we send only ``temperature`` (the knob the caller
    already sets in ``teacher_distill.yaml``). ``top_p`` is accepted but
    currently unused by the Bedrock path — the YAML still carries it so
    the vllm_local path keeps working unchanged.

    Retries on throttling / transient errors with exponential backoff.
    """
    import time
    del top_p  # see docstring; intentionally unused for the bedrock path
    req = {
        "modelId": model_id,
        "messages": [
            {"role": "user", "content": [{"text": user_message}]},
        ],
        "inferenceConfig": {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
        },
    }
    for attempt in range(max_retries):
        try:
            resp = client.converse(**req)
            out_msg = resp.get("output", {}).get("message", {})
            parts = []
            for b in out_msg.get("content", []):
                if isinstance(b, dict) and "text" in b:
                    parts.append(b["text"])
            text = "".join(parts)
            n_tok = int(resp.get("usage", {}).get("outputTokens", 0))
            return text, n_tok, None
        except Exception as e:  # noqa: BLE001
            err = str(e)
            base = 30 if "too many tokens" in err.lower() else (
                4 if "throttl" in err.lower() else 2
            )
            delay = base * (2 ** attempt)
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return None, 0, err
    return None, 0, "exhausted retries"


# ── Config plumbing ─────────────────────────────────────────────────────

def _load_real_synth_config(workspace: Any, stage: dict) -> dict:
    """Collect teacher-distill config from the evolvable YAML + the stage.

    ``teacher_provider`` selects between ``vllm_local`` (load a local
    model on GPU via vLLM) and ``bedrock`` (call AWS Bedrock Converse).
    When provider=bedrock, ``teacher_model_path`` is treated as a
    Bedrock model ID (e.g. ``us.anthropic.claude-sonnet-4-6-v1:0``).
    """
    teacher_cfg_path = Path(workspace.root) / "data" / "generators" / "teacher_distill.yaml"
    teacher_cfg = _load_yaml_safely(teacher_cfg_path)
    base_cfg = _load_yaml_safely(Path(workspace.root) / "model" / "base.yaml")

    outdir = Path(workspace.root) / "data" / "synth"
    out_name = stage.get("out_name") or teacher_cfg.get("out_name", "teacher_traces.jsonl")
    out_path = outdir / out_name

    prompts_csv = teacher_cfg.get("prompts_csv") or os.environ.get(
        "AE_TRAIN_LOCAL_CSV",
        "/fsx/zzsamshi/nemotron-auto-research/data/splits/train_local.csv",
    )

    provider = str(teacher_cfg.get("teacher_provider", "vllm_local"))
    # Per-provider defaults for teacher_model_path when the YAML omits
    # it — so a YAML that only flips provider=bedrock still works.
    default_teacher_path = {
        "vllm_local": "/fsx/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        # us-west-2 Sonnet 4.6 inference profile (per bedrock list-inference-profiles).
        "bedrock": "us.anthropic.claude-sonnet-4-6",
    }.get(provider, "/fsx/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")

    return {
        "teacher_provider": provider,
        "teacher_model_path": teacher_cfg.get(
            "teacher_model_path", default_teacher_path,
        ),
        "bedrock_region": teacher_cfg.get("bedrock_region"),
        "student_model_path": base_cfg.get("path"),
        "prompts_csv": prompts_csv,
        "domains": list(teacher_cfg.get("domains", ["cipher", "bits"])),
        "per_domain": int(teacher_cfg.get("per_domain", 250)),
        "n_samples": int(teacher_cfg.get("n_samples", 1)),
        "min_tokens": int(teacher_cfg.get("min_tokens", 2500)),
        "max_tokens": int(teacher_cfg.get("max_tokens", 8192)),
        "temperature": float(teacher_cfg.get("temperature", 0.8)),
        "top_p": float(teacher_cfg.get("top_p", 0.95)),
        "seed": int(teacher_cfg.get("seed", 17)),
        # vllm_local-only knobs (unused by the bedrock path)
        "tp": int(teacher_cfg.get("tp", 4)),
        "max_num_seqs": int(teacher_cfg.get("max_num_seqs", 16)),
        "gpu_memory_utilization": float(teacher_cfg.get("gpu_memory_utilization", 0.9)),
        "out_path": str(out_path),
        "stage_name": stage.get("name", "teacher_distill"),
    }


def _append_to_sources(workspace: Any, jsonl_path: Path) -> None:
    sources_path = Path(workspace.root) / "data" / "sources.yaml"
    if sources_path.exists():
        with open(sources_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}
    sources = list(raw.get("sources") or [])
    path_str = str(jsonl_path)
    if any(s.get("path") == path_str for s in sources if isinstance(s, dict)):
        return
    sources.append({"path": path_str, "split": "train", "format": "jsonl"})
    raw["sources"] = sources
    with open(sources_path, "w") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)


def _load_yaml_safely(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ── CLI ──────────────────────────────────────────────────────────────────

def _main_cli() -> int:
    parser = argparse.ArgumentParser(description="Run the real synth worker.")
    parser.add_argument("--config", required=True, help="Path to the JSON config dumped by run_synth_stage.")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    _run_real_synth_inproc(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(_main_cli())


# ── StageRegistry adapter ────────────────────────────────────────────────
# Pipeline YAML key is still ``synth_generate`` (user-facing, kept for
# backward compat). The module was renamed; the config string stays.

from ...stage_registry import StageContext, StageResult, register_stage  # noqa: E402


@register_stage("synth_generate")
def _synth_generate_stage_adapter(ctx: StageContext) -> StageResult:
    out_path, stats = run_synth_stage(
        ctx.workspace,
        ctx.stage,
        smoke=ctx.smoke,
        budget_seconds=ctx.budget_seconds,
    )
    return StageResult(
        checkpoint=None,
        metrics={"type": "synth_generate", "out_path": str(out_path), **stats},
    )
