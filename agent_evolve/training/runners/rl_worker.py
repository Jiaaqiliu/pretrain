"""GSPO / DAPO RL stage runner.

Drives the full rollout → advantage → update loop through the TinkerLite
``SamplingClient`` + ``TrainingClient`` protocols, without changing those
protocol signatures. Port of the verified recipe in
``/fsx/zzsamshi/nemotron-auto-research/scripts/gspo_rollout.py`` +
``scripts/gspo_update.py``.

Datum shape for ``loss_fn="gspo"``:
    model_input.tokens = prompt_ids + completion_ids
    loss_fn_inputs = {
        "logprobs_old": List[float],   # per-completion-token
        "advantage":   float,          # group-normalized
        "prompt_len":  int,            # so the client knows where
                                       # the completion starts
    }

The per-cycle entry point is :func:`run_gspo_stage`. It returns
``(CheckpointRef, stage_metrics)`` so it can be swapped into
``SingleNodeTinkerLiteBackend._run_pipeline`` alongside SFT.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ...backends.tinkerlite.base import (
    AdamParams,
    Datum,
    ModelInput,
    Prompt,
    SamplingClient,
    SamplingParams,
    TrainingClient,
)
from ...benchmarks.nemo_reasoner import (
    build_eval_prompt,
    extract_final_answer,
    verify,
)
from ..types import CheckpointRef

logger = logging.getLogger(__name__)


# ── Public entrypoint ────────────────────────────────────────────────────

def run_gspo_stage(
    workspace: Any,
    stage: dict,
    *,
    sampling_client: SamplingClient | None,
    training_client_factory: Callable[[], TrainingClient],
    benchmark: Any,  # noqa: ARG001 — currently uses module-level verify()
    budget_seconds: float | None = None,  # noqa: ARG001 — reserved
    smoke: bool = False,
    training_client: TrainingClient | None = None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Run one GSPO / DAPO RL stage.

    ``training_client_factory`` is a zero-arg callable that builds (and loads
    onto GPU) the :class:`HFTrainingClient` for the update phase. We defer
    its construction until **after** the sampling client is closed, so the
    vLLM engine + HF model never share GPU memory — they'd collectively
    exceed a 1-GPU budget on the 30B base.

    For smoke runs, callers may pass an already-built ``training_client``
    (usually :class:`MockTrainingClient`) instead.
    """
    if smoke:
        assert training_client is not None, "smoke requires a training_client"
        return _run_smoke_gspo(workspace, stage, training_client)
    assert sampling_client is not None, "real GSPO requires a sampling_client"
    return _run_real_gspo(
        workspace,
        stage,
        sampling_client=sampling_client,
        training_client_factory=training_client_factory,
    )


# ── Smoke path (no GPU) ──────────────────────────────────────────────────

def _run_smoke_gspo(
    workspace: Any, stage: dict, training_client: TrainingClient
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Exercise the protocol: no vLLM, no torch.

    Fakes 4 rollouts with synthetic logprobs_old so we can drive
    ``training_client.forward_backward("gspo", ...)`` end-to-end. Used by
    tests + dev loops.
    """
    prompt_ids = [1, 2, 3]
    records = [
        {
            "pid": 0,
            "sid": s,
            "domain": "bits",
            "prompt_ids": prompt_ids,
            "completion_tokens": [10 + s, 11, 12],
            "logprobs_old": [-1.0, -1.0, -1.0],
            "correct": bool(s % 2),
        }
        for s in range(4)
    ]
    group_normalize_advantages(records)
    for r in records:
        datum = Datum(
            model_input=ModelInput.from_ints(
                list(r["prompt_ids"]) + list(r["completion_tokens"])
            ),
            loss_fn_inputs={
                "logprobs_old": list(r["logprobs_old"]),
                "advantage": float(r["advantage"]),
                "prompt_len": len(r["prompt_ids"]),
            },
        )
        training_client.forward_backward(
            [datum], loss_fn="gspo", loss_config={"eps_low": 3e-4, "eps_high": 4e-4}
        )
    training_client.optim_step(AdamParams(learning_rate=1e-5))
    ckpt = training_client.save_weights_for_sampler(stage.get("name", "rl_gspo"))
    return ckpt, {
        "stage": stage.get("name"),
        "total_rollouts": len(records),
        "avg_advantage": sum(r["advantage"] for r in records) / len(records),
        "loss_fn": "gspo",
    }


# ── Real path ────────────────────────────────────────────────────────────

def _run_real_gspo(
    workspace: Any,
    stage: dict,
    *,
    sampling_client: SamplingClient,
    training_client_factory: Callable[[], TrainingClient],
) -> tuple[CheckpointRef, dict[str, Any]]:
    cfg = _load_real_gspo_config(workspace, stage)
    outdir = Path(workspace.root) / "evolution" / "rl" / stage.get("name", "rl_gspo")
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Rollout ───────────────────────────────────────────────────
    # Rollout uses a lightweight tokenizer (no torch model yet) so we don't
    # pull in the training client's weights until after the vLLM engine is
    # torn down. This is what keeps us within a 1-GPU budget on the 30B
    # base model.
    t0 = time.time()
    print(
        f"[gspo] rollout: per_domain={cfg['per_domain']} G={cfg['n_samples']} "
        f"domains={cfg['domains']}"
    )
    prompts_df = _load_train_prompts(cfg)
    from transformers import AutoTokenizer

    model_path = cfg.get("model_path") or _model_path_from_workspace(workspace)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompts: list[Prompt] = []
    prompt_texts: list[str] = []
    per_domain_meta: list[tuple[int, str, str, str]] = []  # (pid, domain, answer, raw_prompt)
    for pid, row in enumerate(prompts_df):
        rendered = build_eval_prompt(row["prompt"], tokenizer)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompt_obj = Prompt(tokens=prompt_ids)
        prompts.append(prompt_obj)
        prompt_texts.append(rendered)
        per_domain_meta.append((pid, row["domain"], str(row["answer"]), row["prompt"]))

    sampling_client.set_prompt_strings(prompts, prompt_texts)
    params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=int(cfg["max_tokens"]),
        n=int(cfg["n_samples"]),
    )
    responses = sampling_client.sample(prompts, params)
    rollout_seconds = time.time() - t0
    print(f"[gspo] rollout done in {rollout_seconds / 60:.1f} min")

    # ── 2. Score + assemble records ──────────────────────────────────
    records: list[dict[str, Any]] = []
    n_correct = 0
    for (pid, domain, answer, raw_prompt), resp in zip(per_domain_meta, responses):
        prompt_ids = prompts[pid].tokens
        for sid, sample in enumerate(resp.samples):
            token_ids = list(sample.tokens)
            lp_old = list(getattr(sample, "_logprobs_per_token", []))
            if len(lp_old) != len(token_ids):
                # vLLM occasionally elides EOS logprobs; pad with mean.
                mean_lp = sample.logprob or -1.0
                lp_old = (lp_old + [mean_lp] * len(token_ids))[: len(token_ids)]
            pred = extract_final_answer(sample.text or "")
            correct = bool(verify(str(answer), str(pred)))
            n_correct += int(correct)
            records.append(
                {
                    "pid": pid,
                    "sid": sid,
                    "domain": domain,
                    "answer": answer,
                    "prompt_raw": raw_prompt,
                    "prompt_ids": list(prompt_ids),
                    "completion_tokens": token_ids,
                    "logprobs_old": lp_old,
                    "correct": correct,
                    "n_tokens": len(token_ids),
                }
            )
    total_rollouts = len(records)
    corr_rate = n_correct / max(1, total_rollouts)
    print(f"[gspo] rollouts={total_rollouts} correctness_rate={corr_rate:.3f}")

    # Persist raw rollouts for later inspection (mirrors gspo_rollout.py).
    rollouts_path = outdir / "rollouts.jsonl"
    with open(rollouts_path, "w") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != "prompt_ids"}) + "\n")

    # ── 3. Advantage (group z-score / loop / domain) ─────────────────
    group_normalize_advantages(
        records,
        advantage_mode=cfg["advantage_mode"],
        length_penalty_lambda=cfg["length_penalty_lambda"],
        length_penalty_cap=cfg["length_penalty_cap"],
    )
    # Drop rollouts with ~zero advantage (group had uniform reward).
    records = [r for r in records if abs(r.get("advantage", 0.0)) > 1e-6]
    # Drop rollouts too long to fit the train budget.
    max_len = int(cfg["max_len"])
    records = [
        r
        for r in records
        if len(r["prompt_ids"]) + len(r["completion_tokens"]) <= max_len
    ]
    print(f"[gspo] non-degenerate, length-ok rollouts: {len(records)}")

    # ── 3.5. Tear down rollout engine *before* building training client.
    # The HF model alone uses ~60 GiB in bf16 + activations; the vLLM engine
    # holds another ~70 GiB. Co-resident on a 140 GiB GPU they OOM during
    # the first backward. Close the sampling client, flush CUDA cache, only
    # then materialize the training client.
    close = getattr(sampling_client, "close", None)
    if close is not None:
        close()
    import gc as _gc
    _gc.collect()
    try:
        import torch as _torch

        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()
    except Exception:
        pass

    # ── DDP path: dispatch GSPO update to a torchrun subprocess ────────
    # AE_TRAIN_DDP=1 enables true data-parallel training across all visible
    # GPUs. The subprocess spawns N ranks, each loads the model on its GPU,
    # shards ``records`` strided, and DDP auto-syncs gradients on backward.
    if os.environ.get("AE_TRAIN_DDP", "0") == "1":
        ddp_ckpt, ddp_stats = _run_gspo_update_ddp(
            workspace, stage, records, cfg, rollouts_path=rollouts_path
        )
        stats = {
            "stage": stage.get("name"),
            "loss_fn": "dapo_token_level" if cfg["dapo_token_level"] else "gspo",
            "total_rollouts": total_rollouts,
            "non_degenerate_rollouts": len(records),
            "rollout_correctness": corr_rate,
            "rollout_seconds": rollout_seconds,
            "advantage_mode": cfg["advantage_mode"],
            "rollouts_path": str(rollouts_path),
            **ddp_stats,
        }
        (outdir / "stats.json").write_text(json.dumps(stats, indent=2))
        return ddp_ckpt, stats

    print("[gspo] building training client (post-rollout)…")
    training_client = training_client_factory()

    # ── 4. Update ────────────────────────────────────────────────────
    rng = random.Random(cfg["seed"])
    rng.shuffle(records)
    grad_accum = int(cfg["grad_accum"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    eps_low = float(cfg["eps_low"])
    eps_high = float(cfg["eps_high"])
    token_level = bool(cfg["dapo_token_level"])
    max_steps = cfg.get("max_steps")

    micro = 0
    opt_steps = 0
    accum_loss = 0.0
    accum_s = 0.0
    accum_clip = 0.0
    t_update = time.time()
    stop = False
    for epoch in range(epochs):
        if stop:
            break
        order = list(range(len(records)))
        rng.shuffle(order)
        for idx in order:
            r = records[idx]
            datum = Datum(
                model_input=ModelInput.from_ints(
                    list(r["prompt_ids"]) + list(r["completion_tokens"])
                ),
                loss_fn_inputs={
                    "logprobs_old": list(r["logprobs_old"]),
                    "advantage": float(r["advantage"]),
                    "prompt_len": len(r["prompt_ids"]),
                },
            )
            result = training_client.forward_backward(
                [datum],
                loss_fn="dapo_token_level" if token_level else "gspo",
                loss_config={
                    "eps_low": eps_low,
                    "eps_high": eps_high,
                    "grad_accum": grad_accum,
                },
            )
            accum_loss += float(result.loss) * grad_accum
            accum_s += float(result.extras.get("mean_s", 0.0))
            accum_clip += float(result.extras.get("clip_frac", 0.0))
            micro += 1
            if micro % grad_accum == 0:
                training_client.optim_step(AdamParams(learning_rate=lr))
                opt_steps += 1
                if opt_steps == 1 or opt_steps % max(1, int(cfg["log_every"])) == 0:
                    el = (time.time() - t_update) / 60.0
                    print(
                        f"  step {opt_steps} loss={accum_loss/grad_accum:.4f} "
                        f"mean_s={accum_s/grad_accum:.4f} "
                        f"clip_frac={accum_clip/grad_accum:.2f} elapsed={el:.1f}min"
                    )
                accum_loss = accum_s = accum_clip = 0.0
                if max_steps is not None and opt_steps >= int(max_steps):
                    stop = True
                    break

    update_seconds = time.time() - t_update

    # ── 5. Save adapter ──────────────────────────────────────────────
    ckpt = training_client.save_weights_for_sampler(stage.get("name", "rl_gspo"))
    print(f"[gspo] saved adapter to {ckpt.path}")

    stats = {
        "stage": stage.get("name"),
        "loss_fn": "dapo_token_level" if token_level else "gspo",
        "total_rollouts": total_rollouts,
        "non_degenerate_rollouts": len(records),
        "rollout_correctness": corr_rate,
        "opt_steps": opt_steps,
        "rollout_seconds": rollout_seconds,
        "update_seconds": update_seconds,
        "lr": lr,
        "eps_low": eps_low,
        "eps_high": eps_high,
        "advantage_mode": cfg["advantage_mode"],
        "rollouts_path": str(rollouts_path),
    }
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2))
    return ckpt, stats


# ── Advantage computation ────────────────────────────────────────────────

def group_normalize_advantages(
    records: list[dict[str, Any]],
    *,
    advantage_mode: str = "group",
    length_penalty_lambda: float = 0.0,
    length_penalty_cap: int = 2500,
) -> None:
    """Port of ``gspo_update.group_normalize_advantages``.

    Mutates each record to set ``reward_combined`` and ``advantage``.
    """
    for r in records:
        rw = 1.0 if r.get("correct") else 0.0
        if length_penalty_lambda > 0.0:
            n_tok = r.get("n_tokens", len(r.get("completion_tokens", [])))
            overage = max(0, n_tok - length_penalty_cap)
            rw -= length_penalty_lambda * overage / 1000.0
        r["reward_combined"] = rw

    if advantage_mode == "domain":
        by_dom: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            by_dom[r.get("domain", "")].append(r)
        for _, g in by_dom.items():
            rewards = [r["reward_combined"] for r in g]
            m = sum(rewards) / len(rewards)
            var = sum((x - m) ** 2 for x in rewards) / len(rewards)
            sd = math.sqrt(var) + 1e-6
            for r, rw in zip(g, rewards):
                r["advantage"] = (rw - m) / sd
        return

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r.get("domain", ""), r["pid"])].append(r)

    for _, g in groups.items():
        rewards = [r["reward_combined"] for r in g]
        if advantage_mode == "loop":
            total = sum(rewards)
            n = len(rewards)
            for r, rw in zip(g, rewards):
                if n > 1:
                    other_mean = (total - rw) / (n - 1)
                    r["advantage"] = rw - other_mean
                else:
                    r["advantage"] = 0.0
        else:  # "group" (z-score within prompt-group)
            m = sum(rewards) / len(rewards)
            var = sum((x - m) ** 2 for x in rewards) / len(rewards)
            sd = math.sqrt(var) + 1e-6
            for r, rw in zip(g, rewards):
                r["advantage"] = (rw - m) / sd


# ── Config + data loading ────────────────────────────────────────────────

def _load_real_gspo_config(workspace: Any, stage: dict) -> dict:
    root = Path(workspace.root)

    def _load_yaml(p: Path) -> dict:
        if not p.exists():
            return {}
        try:
            with open(p) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    optimizer_cfg = _load_yaml(root / "train" / "optimizer.yaml")
    batching_cfg = _load_yaml(root / "train" / "batching.yaml")

    # Prefer the RL-stage knobs embedded in train/pipeline.yaml, fall back
    # to a dedicated rl/gspo.yaml if present.
    rl_yaml = _load_yaml(root / "rl" / "gspo.yaml")

    def _get(key: str, default: Any) -> Any:
        if key in stage:
            return stage[key]
        if key in rl_yaml:
            return rl_yaml[key]
        return default

    prompts_csv = _get(
        "prompts_csv",
        os.environ.get(
            "AE_TRAIN_LOCAL_CSV",
            "/fsx/zzsamshi/nemotron-auto-research/data/splits/train_local.csv",
        ),
    )

    return {
        "prompts_csv": prompts_csv,
        "domains": list(_get("domains", ["bits", "cipher"])),
        "per_domain": int(_get("per_domain", 25)),
        "n_samples": int(_get("n_samples", 4)),
        "max_tokens": int(_get("max_tokens", 2560)),
        "seed": int(_get("seed", 11)),
        "epochs": int(_get("epochs", 1)),
        "grad_accum": int(_get("grad_accum", batching_cfg.get("grad_accum", 8))),
        "lr": float(_get("lr", optimizer_cfg.get("rl_lr", 1e-5))),
        "eps_low": float(_get("eps_low", 3e-4)),
        "eps_high": float(_get("eps_high", 4e-4)),
        "advantage_mode": str(_get("advantage_mode", "group")),
        "length_penalty_lambda": float(_get("length_penalty_lambda", 0.0)),
        "length_penalty_cap": int(_get("length_penalty_cap", 2500)),
        "dapo_token_level": bool(_get("dapo_token_level", False)),
        "max_len": int(_get("max_len", batching_cfg.get("max_seq_len", 2800))),
        "max_steps": _get("max_steps", None),
        "log_every": int(_get("log_every", 4)),
    }


def _model_path_from_workspace(workspace: Any) -> str:
    path = Path(workspace.root) / "model" / "base.yaml"
    if not path.exists():
        raise RuntimeError(f"Missing {path} — cannot resolve base model path")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    model_path = cfg.get("path")
    if not model_path:
        raise RuntimeError(f"{path}::path is empty; GSPO rollout needs the base tokenizer")
    return str(model_path)


def _load_train_prompts(cfg: dict) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(cfg["prompts_csv"])
    sampled: list[dict[str, Any]] = []
    per_domain = int(cfg["per_domain"])
    for dom in cfg["domains"]:
        g = df[df["domain"] == dom]
        k = min(per_domain, len(g))
        if k == 0:
            continue
        picked = g.sample(n=k, random_state=cfg["seed"])
        for _, row in picked.iterrows():
            sampled.append(
                {
                    "prompt": str(row["prompt"]),
                    "answer": str(row["answer"]),
                    "domain": str(row["domain"]),
                }
            )
    return sampled


# ── DDP dispatch for the GSPO update phase ──────────────────────────────

def _run_gspo_update_ddp(
    workspace: Any,
    stage: dict,
    records: list[dict[str, Any]],
    cfg: dict,
    *,
    rollouts_path: Path,
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Write post-advantage records to disk, launch torchrun DDP worker."""
    from ...backends.tinkerlite.single_node.ddp_launcher import run_gspo_ddp

    outdir = Path(workspace.root) / "evolution" / "rl" / stage.get("name", "rl_gspo")
    outdir.mkdir(parents=True, exist_ok=True)

    # Dump the post-advantage record set (already filtered for non-zero
    # advantage + length). One JSONL line per rollout.
    records_path = outdir / "records_with_advantage.jsonl"
    with open(records_path, "w") as f:
        for r in records:
            # Strip anything huge we don't need in the subprocess; keep the
            # fields the DDP worker reads (prompt_ids, completion_tokens,
            # logprobs_old, advantage).
            keep = {
                "prompt_ids": list(r["prompt_ids"]),
                "completion_tokens": list(r["completion_tokens"]),
                "logprobs_old": list(r["logprobs_old"]),
                "advantage": float(r["advantage"]),
                "pid": r.get("pid"),
                "domain": r.get("domain"),
            }
            f.write(json.dumps(keep) + "\n")
    print(f"[gspo-ddp] wrote {len(records)} records to {records_path}")

    # Load base + adapter YAML snapshots the launcher needs.
    root = Path(workspace.root)

    def _load(rel: str) -> dict[str, Any]:
        path = root / rel
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    base_cfg = _load("model/base.yaml")
    adapter_cfg = _load("model/adapter.yaml")
    optimizer_cfg = _load("train/optimizer.yaml")

    start_adapter = adapter_cfg.get("seed_adapter_path")
    if not start_adapter:
        raise RuntimeError(
            "GSPO DDP needs model/adapter.yaml::seed_adapter_path to be set "
            "(the starting-policy adapter for the update)."
        )

    gspo_cfg = {
        "epochs": cfg["epochs"],
        "grad_accum": cfg["grad_accum"],
        "lr": cfg["lr"],
        "eps_low": cfg["eps_low"],
        "eps_high": cfg["eps_high"],
        "dapo_token_level": cfg["dapo_token_level"],
        "max_steps": cfg.get("max_steps"),
        "log_every": cfg["log_every"],
        "seed": cfg["seed"],
    }
    return run_gspo_ddp(
        workspace,
        stage,
        base_cfg=base_cfg,
        adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg,
        rollouts_path=records_path,
        start_adapter_path=str(start_adapter),
        gspo_cfg=gspo_cfg,
    )


__all__ = ["run_gspo_stage", "group_normalize_advantages"]
