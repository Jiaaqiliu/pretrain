"""nemo_mas trainer CLI — thin Bash-facing shim over existing handlers.

Built so Claude Code skills can drive training via plain shell instead of
MCP tools. Every subcommand returns a single JSON object on stdout and
exits 0 on handler success (``"ok": true``), exit 1 on handler-level
error (``"ok": false, "reason": ...``), exit 2 on argparse-level error.

Env vars (must be set before calling):

  * ``NEMO_MAS_WORK_DIR``        — run root, determines default memory path.
  * ``NEMO_MAS_WORKSPACE_ROOT``  — forked workspace for current cycle.
  * ``NEMO_MAS_MEMORY_PATH``     — overrides the default records.jsonl path.
  * ``NEMO_MAS_COMPUTE_BACKEND`` — ``k8s`` / ``local`` / ``none``. Training
                                    subcommands require this to be set.

Subcommands:

  train launch    --recipe P --data P --out P [--max-steps N] [--monitor]
  train cancel    (--job-name N | --name-contains N) [--force]
  pack            --ckpt P --out ZIP
  log tail        --job-id ID [--lines N]
  metric read     --ckpt P
  stability       --ids REC_ID [--ids REC_ID ...]   # reads memory
  k8s status      [--job-name N] [--name-contains S]
  mem append      --role R --kind K --title T --body-file F [--ref R ...] [--tag T ...]
  mem get         --id REC_ID
  mem search      --query Q [--kind K] [--top-k N]
  mem recent      [--kind K] [-k N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .agent_teams.hook_utils import (
    current_checkpoint_mode,
    current_memory_path,
    current_workspace_root,
)
from .backends import BackendBridge, local_handlers
from .checkpoints import (
    CHECKPOINT_MODE_AUTO,
    CHECKPOINT_MODE_MANUAL,
    VALID_VERDICTS,
    fold_checkpoints,
    load_slot_decls,
)
from .memory import RecipeMemory
from .schema import RecordValidationError


def _emit(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, default=str))
    return 0 if payload.get("ok") else 1


def _ws_resolver():
    return current_workspace_root() or Path(
        os.environ.get("NEMO_MAS_SEED_WORKSPACE", "seed_workspaces/nemo_mas_reasoner")
    )


class _CliError(Exception):
    """Handler-level error surfaced as ``{"ok": false, "reason": ...}``."""


def _memory() -> RecipeMemory:
    path = current_memory_path()
    if path is None:
        raise _CliError(
            "NEMO_MAS_MEMORY_PATH (or NEMO_MAS_WORK_DIR) must be set before "
            "calling `mem` / `stability` subcommands."
        )
    return RecipeMemory(path)


def _require_bridge() -> BackendBridge:
    bridge = BackendBridge.from_env(_ws_resolver)
    if bridge is None:
        raise _CliError(
            "NEMO_MAS_COMPUTE_BACKEND must be set to 'k8s' or 'local' for "
            "training subcommands. Set it then re-run."
        )
    return bridge


# ── train ────────────────────────────────────────────────────────────


def _cmd_train_launch(args: argparse.Namespace) -> int:
    bridge = _require_bridge()
    out = bridge.launch_training(
        recipe_path=args.recipe,
        data_path=args.data,
        ckpt_out=args.out,
        max_steps=args.max_steps,
        monitor=args.monitor,
    )
    return _emit(json.loads(out))


def _cmd_train_cancel(args: argparse.Namespace) -> int:
    bridge = _require_bridge()
    out = bridge.cancel_training(
        job_name=args.job_name,
        name_contains=args.name_contains,
        stuck_only=not args.force,
    )
    return _emit(json.loads(out))


# ── single-shot local handlers ────────────────────────────────────────


def _cmd_pack(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["pack_submission"](ckpt_path=args.ckpt, out_zip=args.out)
    return _emit(json.loads(out))


def _cmd_log_tail(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["read_training_log"](job_id=args.job_id, tail_lines=args.lines)
    return _emit(json.loads(out))


def _cmd_metric_read(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["read_checkpoint_metric"](ckpt_path=args.ckpt)
    return _emit(json.loads(out))


def _cmd_k8s_status(args: argparse.Namespace) -> int:
    bridge = _require_bridge()
    out = bridge.k8s_status(
        job_name=args.job_name, name_contains=args.name_contains
    )
    return _emit(json.loads(out))


# ── stability: fold metrics across training_run records ───────────────


def _cmd_stability(args: argparse.Namespace) -> int:
    mem = _memory()
    per_id: dict[str, float | None] = {}
    missing: list[str] = []
    for rid in args.ids:
        rec = mem.get(rid)
        if rec is None:
            missing.append(rid)
            continue
        metric = _extract_primary_metric(rec.body)
        per_id[rid] = metric
    numeric = [v for v in per_id.values() if isinstance(v, (int, float))]
    if not numeric or missing:
        return _emit({
            "ok": False,
            "reason": (
                f"missing_ids={missing}" if missing
                else "no numeric primary_metric found in any training_run body"
            ),
            "per_id": per_id,
        })
    import statistics
    mean = statistics.fmean(numeric)
    stdev = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
    rel = (stdev / mean) if mean else 0.0
    return _emit({
        "ok": True,
        "per_id": per_id,
        "mean": mean,
        "stdev": stdev,
        "rel_stdev": rel,
        "n": len(numeric),
    })


def _extract_primary_metric(body: str) -> float | None:
    """Extract the primary eval metric from a training_run body.

    Search order:
      1. A fenced ```json block with key ``primary_metric_value`` or
         nested ``metrics.kaggle`` / ``metrics.local``.
      2. A line matching ``primary_metric_value: <number>``.
    Returns None if nothing plausible is found.
    """
    import re
    fence_re = re.compile(r"```json\s*(.*?)```", re.DOTALL)
    for m in fence_re.finditer(body):
        try:
            d = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            for key in ("primary_metric_value", "metric_value", "score"):
                if isinstance(d.get(key), (int, float)):
                    return float(d[key])
            metrics = d.get("metrics")
            if isinstance(metrics, dict):
                for key in ("kaggle", "local", "hard"):
                    v = metrics.get(key)
                    if isinstance(v, (int, float)):
                        return float(v)
    line_re = re.compile(r"primary_metric_value\s*[:=]\s*([+-]?\d+(?:\.\d+)?)")
    m = line_re.search(body)
    if m:
        return float(m.group(1))
    return None


# ── planner: recipe yaml diff ─────────────────────────────────────────


def _cmd_recipe_diff(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["diff_yaml"](a=args.a, b=args.b)
    return _emit(json.loads(out))


# ── data worker: filter / dedup / mix / write ─────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _cmd_data_filter_by_gold(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    gens_p = Path(args.generations)
    golds_p = Path(args.golds)
    if not gens_p.is_file():
        return _emit({"ok": False, "reason": f"generations file not found: {gens_p}"})
    if not golds_p.is_file():
        return _emit({"ok": False, "reason": f"golds file not found: {golds_p}"})
    generations = _load_jsonl(gens_p)
    golds_rows = _load_jsonl(golds_p)
    if args.gold_field:
        golds = [r.get(args.gold_field) for r in golds_rows]
    else:
        golds = golds_rows  # assume golds file is a JSONL of plain scalars as rows? Reject.
        return _emit({"ok": False,
                      "reason": "--gold-field must name the field in --golds holding the gold answer"})
    out = handlers["filter_by_gold"](generations=generations, golds=golds)
    result = json.loads(out)
    # If the caller asked for a written output, dump `kept` to JSONL.
    if result.get("ok") and args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            for row in result["kept"]:
                f.write(json.dumps(row) + "\n")
        result["output_path"] = str(out_p)
        # Drop the inline kept rows to keep stdout small.
        result.pop("kept", None)
    return _emit(result)


def _cmd_data_dedup(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["minhash_dedup"](
        input_path=args.path, key_field=args.key_field, threshold=args.threshold,
    )
    return _emit(json.loads(out))


def _cmd_data_format_filter(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["apply_format_filter"](input_path=args.path)
    return _emit(json.loads(out))


def _cmd_data_mix(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    if len(args.source) != len(args.weight):
        return _emit({"ok": False,
                      "reason": "--source count must equal --weight count"})
    out = handlers["mix_sources"](
        sources=list(args.source), weights=list(args.weight),
        curriculum_yaml=args.curriculum,
    )
    return _emit(json.loads(out))


def _cmd_data_write(args: argparse.Namespace) -> int:
    """Copy a JSONL file (possibly transformed) into a workspace-safe path."""
    handlers = dict(local_handlers(_ws_resolver))
    src = Path(args.from_)
    if not src.is_file():
        return _emit({"ok": False, "reason": f"--from file not found: {src}"})
    rows = _load_jsonl(src)
    out = handlers["write_jsonl"](path=args.path, rows=rows)
    return _emit(json.loads(out))


# ── teacher distill (data worker) ─────────────────────────────────────


def _cmd_teacher_call(args: argparse.Namespace) -> int:
    bridge = _require_bridge()
    prompts_p = Path(args.prompts)
    if not prompts_p.is_file():
        return _emit({"ok": False, "reason": f"--prompts file not found: {prompts_p}"})
    rows = _load_jsonl(prompts_p)
    if args.prompt_field:
        prompts = [r.get(args.prompt_field, "") for r in rows]
    else:
        return _emit({"ok": False,
                      "reason": "--prompt-field must name the field in --prompts holding the prompt text"})
    system_prompt = None
    if args.system_prompt_file:
        sp_p = Path(args.system_prompt_file)
        if not sp_p.is_file():
            return _emit({"ok": False,
                          "reason": f"--system-prompt-file not found: {sp_p}"})
        system_prompt = sp_p.read_text(encoding="utf-8")
    out = bridge.call_teacher_model(
        model=args.model, prompts=prompts,
        max_tokens=args.max_tokens, temperature=args.temperature,
        system_prompt=system_prompt,
    )
    result = json.loads(out)
    if result.get("ok") and args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        gens = result.get("generations", [])
        merged = []
        for row, gen in zip(rows, gens):
            merged_row = dict(row)
            merged_row["completion"] = gen if isinstance(gen, str) else gen.get("completion", "")
            merged.append(merged_row)
        with out_p.open("w", encoding="utf-8") as f:
            for row in merged:
                f.write(json.dumps(row) + "\n")
        result["output_path"] = str(out_p)
        result.pop("generations", None)
    return _emit(result)


# ── inference (solver self-distill) ───────────────────────────────────


def _cmd_infer_generate(args: argparse.Namespace) -> int:
    """Combined load_checkpoint_for_inference + batch_generate.

    Bash can't carry the handle across CLI invocations, so we load, generate,
    and dump in one shot.
    """
    bridge = _require_bridge()
    prompts_p = Path(args.prompts)
    if not prompts_p.is_file():
        return _emit({"ok": False, "reason": f"--prompts file not found: {prompts_p}"})
    rows = _load_jsonl(prompts_p)
    if not args.prompt_field:
        return _emit({"ok": False,
                      "reason": "--prompt-field must name the field in --prompts holding the prompt text"})
    prompts = [r.get(args.prompt_field, "") for r in rows]

    load_out = json.loads(bridge.load_checkpoint_for_inference(ckpt_path=args.ckpt))
    if not load_out.get("ok"):
        return _emit(load_out)
    handle = load_out["handle"]

    sampling_config = {}
    if args.temperature is not None:
        sampling_config["temperature"] = args.temperature
    if args.top_p is not None:
        sampling_config["top_p"] = args.top_p
    if args.max_tokens is not None:
        sampling_config["max_tokens"] = args.max_tokens

    gen_out = json.loads(bridge.batch_generate(
        handle=handle, prompts=prompts,
        sampling_config=sampling_config or None,
    ))
    if not gen_out.get("ok"):
        return _emit(gen_out)

    generations = gen_out.get("generations", [])
    merged = []
    for row, gen in zip(rows, generations):
        merged_row = dict(row)
        merged_row["completion"] = gen if isinstance(gen, str) else gen.get("completion", "")
        merged.append(merged_row)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")
    return _emit({
        "ok": True, "output_path": str(out_p), "n": len(merged),
        "ckpt_path": args.ckpt,
        "sampling_config": gen_out.get("sampling_config"),
    })


# ── data audit (reviewer) ─────────────────────────────────────────────


def _cmd_data_sample(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["sample_jsonl"](path=args.path, n=args.n, seed=args.seed)
    return _emit(json.loads(out))


def _cmd_data_count_by(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["count_by_field"](path=args.path, field=args.field)
    return _emit(json.loads(out))


def _cmd_data_length_dist(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["length_distribution"](
        path=args.path, field=args.field, tokenizer=args.tokenizer,
    )
    return _emit(json.loads(out))


def _cmd_data_validate(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["format_validate"](path=args.path)
    return _emit(json.loads(out))


# ── eval (reviewer) ───────────────────────────────────────────────────


def _cmd_eval_run(args: argparse.Namespace) -> int:
    bridge = _require_bridge()
    out = bridge.run_eval(
        ckpt_path=args.ckpt, split=args.split, limit=args.limit,
    )
    return _emit(json.loads(out))


# ── kaggle (reviewer, gated by budget hook) ───────────────────────────


def _cmd_kaggle_submit(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["kaggle_submit"](zip_path=args.zip, message=args.message)
    return _emit(json.loads(out))


def _cmd_kaggle_fetch_score(args: argparse.Namespace) -> int:
    handlers = dict(local_handlers(_ws_resolver))
    out = handlers["kaggle_fetch_score"](submission_id=args.submission_id or "")
    return _emit(json.loads(out))


# ── checkpoints (reviewer QA officer) ─────────────────────────────────


def _fold_current() -> tuple[list, str, list[dict]]:
    ws = current_workspace_root()
    mode = current_checkpoint_mode()
    slots = load_slot_decls(ws) if ws else []
    if not slots:
        return ([], mode, [])
    mem = _memory()
    folded = fold_checkpoints(mem.all_records(), mode, slots=slots)
    return (folded, mode, slots)


def _cmd_ckpt_list(args: argparse.Namespace) -> int:
    folded, mode, _ = _fold_current()
    if not folded:
        return _emit({"ok": True, "slots": [], "mode": mode,
                      "note": "no checkpoints.yaml in active workspace"})
    return _emit({
        "ok": True, "mode": mode,
        "slots": [{
            "id": s.id, "title": s.title, "state": s.state,
            "required": s.required,
            "requires_evidence": list(s.requires_evidence),
            "depends_on": list(s.depends_on),
            "evidence_counts": s.evidence_counts,
            "last_review_verdict": s.last_review_verdict,
            "last_review_reason": s.last_review_reason,
            "can_sign": s.can_sign,
        } for s in folded],
    })


def _cmd_ckpt_state(args: argparse.Namespace) -> int:
    folded, mode, _ = _fold_current()
    for s in folded:
        if s.id == args.slot_id:
            return _emit({"ok": True, "mode": mode, "slot": {
                "id": s.id, "title": s.title, "state": s.state,
                "required": s.required,
                "requires_evidence": list(s.requires_evidence),
                "depends_on": list(s.depends_on),
                "evidence_counts": s.evidence_counts,
                "last_review_verdict": s.last_review_verdict,
                "last_review_reason": s.last_review_reason,
                "can_sign": s.can_sign,
            }})
    return _emit({"ok": False, "reason": f"unknown slot_id={args.slot_id!r}"})


def _cmd_ckpt_review_suggest(args: argparse.Namespace) -> int:
    _, _, slots = _fold_current()
    slot_ids = {s["id"] for s in slots}
    if args.slot_id not in slot_ids:
        return _emit({"ok": False,
                      "reason": f"unknown slot_id={args.slot_id!r}; "
                                f"expected one of {sorted(slot_ids)}"})
    if args.verdict not in VALID_VERDICTS:
        return _emit({"ok": False,
                      "reason": f"verdict must be one of {sorted(VALID_VERDICTS)}; "
                                f"got {args.verdict!r}"})
    if not args.ref:
        return _emit({"ok": False,
                      "reason": "at least one --ref to the evidence being judged"})
    if not args.reason.strip():
        return _emit({"ok": False, "reason": "--reason must be non-empty"})

    combined_tags = [
        f"checkpoint:{args.slot_id}",
        f"verdict:{args.verdict}",
        "channel:qa_review",
    ]
    if args.tag:
        combined_tags.extend(t for t in args.tag if t not in combined_tags)
    mem = _memory()
    try:
        rec = mem.write(
            role="reviewer",
            kind="checkpoint_review",
            title=f"{args.verdict} · {args.slot_id}",
            body=args.reason,
            tags=tuple(combined_tags),
            refs=tuple(args.ref),
        )
    except RecordValidationError as e:
        return _emit({"ok": False, "reason": str(e)})
    return _emit({"ok": True, "id": rec.id,
                  "slot_id": args.slot_id, "verdict": args.verdict})


def _cmd_ckpt_sign(args: argparse.Namespace) -> int:
    folded, mode, slots = _fold_current()
    slots_by_id = {s["id"]: s for s in slots}
    slot = slots_by_id.get(args.slot_id)
    if slot is None:
        return _emit({"ok": False, "reason": f"unknown slot_id={args.slot_id!r}"})
    if mode == CHECKPOINT_MODE_MANUAL and args.role != "human":
        return _emit({"ok": False,
                      "reason": f"mode={mode!r}: only role='human' may sign in "
                                f"manual mode; got role={args.role!r}. Reviewer "
                                f"should post verdict=ready_to_sign and wait."})
    if not args.ref:
        return _emit({"ok": False,
                      "reason": f"sign requires evidence refs covering "
                                f"{slot['requires_evidence']}"})

    mem = _memory()
    ref_kinds: list[str] = []
    for rid in args.ref:
        rec = mem.get(rid)
        if rec is None:
            return _emit({"ok": False, "reason": f"ref {rid!r} does not resolve"})
        ref_kinds.append(rec.kind)
    missing = [k for k in slot["requires_evidence"] if k not in ref_kinds]
    if missing:
        return _emit({"ok": False,
                      "reason": f"slot {args.slot_id!r} requires evidence of kinds "
                                f"{slot['requires_evidence']}; refs cover "
                                f"{ref_kinds}; missing {missing}"})
    folded_by_id = {s.id: s for s in folded}
    unmet = [d for d in slot["depends_on"]
             if folded_by_id.get(d) is None
             or folded_by_id[d].state not in {"signed", "reopened"}]
    if unmet:
        return _emit({"ok": False,
                      "reason": f"slot {args.slot_id!r} depends on {unmet} which "
                                "are not yet signed"})

    # Map reviewer/human/orchestrator_auto → signing-author role.
    if args.role == "human":
        signer_role, actor_label = "orchestrator_auto", "human:lead"
    elif args.role == "orchestrator_auto":
        signer_role, actor_label = "orchestrator_auto", "orchestrator"
    elif args.role == "reviewer":
        signer_role, actor_label = "reviewer", "reviewer"
    else:
        return _emit({"ok": False,
                      "reason": f"--role must be one of human/reviewer/orchestrator_auto; "
                                f"got {args.role!r}"})

    body = json.dumps({
        "checkpoint_id": args.slot_id, "event": "signoff",
        "actor": actor_label, "note": args.note or "",
    })
    try:
        rec = mem.write(
            role=signer_role,
            kind="checkpoint_event",
            title=f"signoff {args.slot_id}",
            body=body,
            tags=(f"checkpoint:{args.slot_id}", "event:signoff",
                  f"actor:{actor_label}"),
            refs=tuple(args.ref),
        )
    except RecordValidationError as e:
        return _emit({"ok": False, "reason": str(e)})
    return _emit({"ok": True, "id": rec.id, "slot_id": args.slot_id,
                  "actor": actor_label, "mode": mode})


# ── mem ──────────────────────────────────────────────────────────────


def _cmd_mem_append(args: argparse.Namespace) -> int:
    mem = _memory()
    body_path = Path(args.body_file)
    if not body_path.is_file():
        return _emit({"ok": False, "reason": f"body_file not found: {body_path}"})
    body = body_path.read_text(encoding="utf-8")
    try:
        rec = mem.write(
            role=args.role,
            kind=args.kind,
            title=args.title,
            body=body,
            tags=args.tag or [],
            refs=args.ref or [],
        )
    except RecordValidationError as e:
        return _emit({"ok": False, "reason": str(e)})
    return _emit({"ok": True, "record": rec.to_dict()})


def _cmd_mem_get(args: argparse.Namespace) -> int:
    rec = _memory().get(args.id)
    if rec is None:
        return _emit({"ok": False, "reason": f"no record with id={args.id!r}"})
    return _emit({"ok": True, "record": rec.to_dict()})


def _cmd_mem_search(args: argparse.Namespace) -> int:
    hits = _memory().search(
        args.query, kind=args.kind, top_k=args.top_k,
    )
    return _emit({
        "ok": True,
        "hits": [
            {"score": round(s, 4), **rec.to_dict()} for rec, s in hits
        ],
    })


def _cmd_mem_recent(args: argparse.Namespace) -> int:
    recs = _memory().recent(kind=args.kind, k=args.k)
    return _emit({"ok": True, "records": [r.to_dict() for r in recs]})


# ── argparse wiring ───────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nemo-mas",
        description="Trainer-facing CLI for nemo_mas (JSON on stdout).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    train = sub.add_parser("train").add_subparsers(dest="train_cmd", required=True)
    tlaunch = train.add_parser("launch")
    tlaunch.add_argument("--recipe", required=True)
    tlaunch.add_argument("--data", required=True)
    tlaunch.add_argument("--out", required=True, help="ckpt output dir")
    tlaunch.add_argument("--max-steps", type=int, default=None)
    tlaunch.add_argument("--monitor", action="store_true", default=True)
    tlaunch.set_defaults(func=_cmd_train_launch)

    tcancel = train.add_parser("cancel")
    g = tcancel.add_mutually_exclusive_group(required=True)
    g.add_argument("--job-name")
    g.add_argument("--name-contains")
    tcancel.add_argument("--force", action="store_true",
                         help="cancel even jobs that are NOT stuck")
    tcancel.set_defaults(func=_cmd_train_cancel)

    # pack
    pack = sub.add_parser("pack")
    pack.add_argument("--ckpt", required=True)
    pack.add_argument("--out", required=True, help="output zip path")
    pack.set_defaults(func=_cmd_pack)

    # log tail
    log = sub.add_parser("log").add_subparsers(dest="log_cmd", required=True)
    ltail = log.add_parser("tail")
    ltail.add_argument("--job-id", required=True)
    ltail.add_argument("--lines", type=int, default=200)
    ltail.set_defaults(func=_cmd_log_tail)

    # metric read
    metric = sub.add_parser("metric").add_subparsers(dest="metric_cmd", required=True)
    mread = metric.add_parser("read")
    mread.add_argument("--ckpt", required=True)
    mread.set_defaults(func=_cmd_metric_read)

    # stability
    stab = sub.add_parser("stability")
    stab.add_argument("--ids", action="append", required=True,
                      help="training_run record id; repeat for each seed")
    stab.set_defaults(func=_cmd_stability)

    # k8s status
    k8s = sub.add_parser("k8s").add_subparsers(dest="k8s_cmd", required=True)
    kstat = k8s.add_parser("status")
    kstat.add_argument("--job-name", default=None)
    kstat.add_argument("--name-contains", default="aev-")
    kstat.set_defaults(func=_cmd_k8s_status)

    # data — audit (reviewer) + curation (data_worker)
    data = sub.add_parser("data").add_subparsers(dest="data_cmd", required=True)
    # audit
    dsamp = data.add_parser("sample")
    dsamp.add_argument("--path", required=True)
    dsamp.add_argument("-n", type=int, default=50)
    dsamp.add_argument("--seed", type=int, default=0)
    dsamp.set_defaults(func=_cmd_data_sample)

    dcb = data.add_parser("count-by")
    dcb.add_argument("--path", required=True)
    dcb.add_argument("--field", required=True)
    dcb.set_defaults(func=_cmd_data_count_by)

    dld = data.add_parser("length-dist")
    dld.add_argument("--path", required=True)
    dld.add_argument("--field", required=True)
    dld.add_argument("--tokenizer", default="approx")
    dld.set_defaults(func=_cmd_data_length_dist)

    dval = data.add_parser("validate")
    dval.add_argument("--path", required=True)
    dval.set_defaults(func=_cmd_data_validate)

    # curation (data_worker)
    dfbg = data.add_parser("filter-by-gold")
    dfbg.add_argument("--generations", required=True,
                      help="JSONL of {completion, ...} rows")
    dfbg.add_argument("--golds", required=True,
                      help="JSONL of rows carrying the gold answer")
    dfbg.add_argument("--gold-field", required=True,
                      help="name of the field in --golds holding the gold")
    dfbg.add_argument("--out", default=None,
                      help="optional JSONL path to write kept rows")
    dfbg.set_defaults(func=_cmd_data_filter_by_gold)

    ddd = data.add_parser("dedup")
    ddd.add_argument("--path", required=True, help="input JSONL (workspace-relative)")
    ddd.add_argument("--key-field", required=True)
    ddd.add_argument("--threshold", type=float, default=0.85)
    ddd.set_defaults(func=_cmd_data_dedup)

    dff = data.add_parser("format-filter")
    dff.add_argument("--path", required=True)
    dff.set_defaults(func=_cmd_data_format_filter)

    dmix = data.add_parser("mix")
    dmix.add_argument("--source", action="append", required=True,
                      help="repeatable; workspace-relative source JSONL")
    dmix.add_argument("--weight", action="append", type=float, required=True,
                      help="repeatable; weight paired with each --source")
    dmix.add_argument("--curriculum", default=None,
                      help="optional curriculum.yaml path (provenance hint only)")
    dmix.set_defaults(func=_cmd_data_mix)

    dwr = data.add_parser("write")
    dwr.add_argument("--from", dest="from_", required=True,
                     help="source JSONL path (absolute or /tmp)")
    dwr.add_argument("--path", required=True,
                     help="workspace-relative destination")
    dwr.set_defaults(func=_cmd_data_write)

    # recipe (planner)
    rec = sub.add_parser("recipe").add_subparsers(dest="recipe_cmd", required=True)
    rdiff = rec.add_parser("diff")
    rdiff.add_argument("--a", required=True,
                       help="YAML path (workspace-relative) OR inline YAML text")
    rdiff.add_argument("--b", required=True,
                       help="YAML path (workspace-relative) OR inline YAML text")
    rdiff.set_defaults(func=_cmd_recipe_diff)

    # teacher distill (data_worker)
    teacher = sub.add_parser("teacher").add_subparsers(
        dest="teacher_cmd", required=True,
    )
    tcall = teacher.add_parser("call")
    tcall.add_argument("--model", required=True)
    tcall.add_argument("--prompts", required=True, help="JSONL of prompt rows")
    tcall.add_argument("--prompt-field", required=True,
                       help="name of the field in --prompts holding prompt text")
    tcall.add_argument("--max-tokens", type=int, default=8000)
    tcall.add_argument("--temperature", type=float, default=0.7)
    tcall.add_argument("--system-prompt-file", default=None)
    tcall.add_argument("--out", default=None,
                       help="optional JSONL to write {row..., completion}")
    tcall.set_defaults(func=_cmd_teacher_call)

    # self-distill (data_worker)
    infer = sub.add_parser("infer").add_subparsers(
        dest="infer_cmd", required=True,
    )
    igen = infer.add_parser("generate")
    igen.add_argument("--ckpt", required=True)
    igen.add_argument("--prompts", required=True)
    igen.add_argument("--prompt-field", required=True)
    igen.add_argument("--out", required=True,
                      help="JSONL output path ({row..., completion})")
    igen.add_argument("--temperature", type=float, default=None)
    igen.add_argument("--top-p", type=float, default=None)
    igen.add_argument("--max-tokens", type=int, default=None)
    igen.set_defaults(func=_cmd_infer_generate)

    # eval (reviewer)
    ev = sub.add_parser("eval").add_subparsers(dest="eval_cmd", required=True)
    erun = ev.add_parser("run")
    erun.add_argument("--ckpt", required=True)
    erun.add_argument("--split", required=True)
    erun.add_argument("--limit", type=int, default=None)
    erun.set_defaults(func=_cmd_eval_run)

    # kaggle (reviewer)
    kg = sub.add_parser("kaggle").add_subparsers(dest="kaggle_cmd", required=True)
    ksub = kg.add_parser("submit")
    ksub.add_argument("--zip", required=True, help="path to submission.zip")
    ksub.add_argument("--message", required=True)
    ksub.set_defaults(func=_cmd_kaggle_submit)

    kfet = kg.add_parser("fetch-score")
    kfet.add_argument("--submission-id", default=None)
    kfet.set_defaults(func=_cmd_kaggle_fetch_score)

    # checkpoints (reviewer)
    ck = sub.add_parser("checkpoints").add_subparsers(
        dest="ckpt_cmd", required=True,
    )
    ckls = ck.add_parser("list")
    ckls.set_defaults(func=_cmd_ckpt_list)

    ckst = ck.add_parser("state")
    ckst.add_argument("--slot-id", required=True)
    ckst.set_defaults(func=_cmd_ckpt_state)

    ckrv = ck.add_parser("review-suggest")
    ckrv.add_argument("--slot-id", required=True)
    ckrv.add_argument("--verdict", required=True,
                      help="one of evidence_attached, ready_to_sign, "
                           "insufficient, reject")
    ckrv.add_argument("--reason", required=True,
                      help="one-line reason shown in the cockpit")
    ckrv.add_argument("--ref", action="append", required=True,
                      help="evidence record id; repeatable")
    ckrv.add_argument("--tag", action="append", default=None)
    ckrv.set_defaults(func=_cmd_ckpt_review_suggest)

    cksg = ck.add_parser("sign")
    cksg.add_argument("--slot-id", required=True)
    cksg.add_argument("--role", default="human",
                      help="human (default, manual or auto), reviewer "
                           "(auto only), orchestrator_auto (auto only)")
    cksg.add_argument("--ref", action="append", required=True,
                      help="evidence ref; repeatable; must cover all "
                           "requires_evidence kinds for the slot")
    cksg.add_argument("--note", default="")
    cksg.set_defaults(func=_cmd_ckpt_sign)

    # mem
    mem = sub.add_parser("mem").add_subparsers(dest="mem_cmd", required=True)
    mapp = mem.add_parser("append")
    mapp.add_argument("--role", required=True,
                      help="must equal 'trainer' for this agent")
    mapp.add_argument("--kind", required=True)
    mapp.add_argument("--title", required=True)
    mapp.add_argument("--body-file", required=True,
                      help="path to a text file with the record body")
    mapp.add_argument("--ref", action="append", default=None,
                      help="referenced record id; repeatable")
    mapp.add_argument("--tag", action="append", default=None,
                      help="tag; repeatable")
    mapp.set_defaults(func=_cmd_mem_append)

    mget = mem.add_parser("get")
    mget.add_argument("--id", required=True)
    mget.set_defaults(func=_cmd_mem_get)

    msearch = mem.add_parser("search")
    msearch.add_argument("--query", required=True)
    msearch.add_argument("--kind", default=None)
    msearch.add_argument("--top-k", type=int, default=8)
    msearch.set_defaults(func=_cmd_mem_search)

    mrecent = mem.add_parser("recent")
    mrecent.add_argument("--kind", default=None)
    mrecent.add_argument("-k", type=int, default=10)
    mrecent.set_defaults(func=_cmd_mem_recent)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except _CliError as e:
        return _emit({"ok": False, "reason": str(e)})
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        return _emit({"ok": False, "reason": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    sys.exit(main())
