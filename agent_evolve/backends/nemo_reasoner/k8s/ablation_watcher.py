#!/usr/bin/env python3
"""ablation_watcher — auto-dispatch eval Jobs once an ablation's train arms finish.

Sister of ``eval_watcher.py``. Polls ``<work_dir>/.pending_jobs/ablation-*.json``
every POLL_SEC seconds and drives the phase-1 → phase-2 transition:

  phase=awaiting_train (or absent):
    - check both arms' train Jobs via kubectl.
    - either still running → leave alone.
    - either failed       → log and skip; let an LLM trainer agent
                            harvest the failure as `failed_attempt`
                            via `trainer-ablation-collect`.
    - both succeeded      → write a `training_run` ledger record per
                            arm (skipping arm A if `arm_a.reused=true`),
                            submit one eval Job per arm that doesn't
                            already have an `eval_report_id`, drop the
                            per-arm eval markers (kind=`eval_report`,
                            same shape as `trainer-run-eval` writes),
                            and flip the ablation marker to
                            phase=awaiting_evals with the new ids.

  phase=awaiting_evals:
    - log "ready for finalization" and leave the marker alone. The
      final `ablation_report` (computing delta, picking verdict,
      narrating side-effects) is intentionally LLM-driven via
      `trainer-ablation-collect` — too much synthesis for shell.

Run via tmux::

    tmux new -d -s ablation_watcher \\
      'NEMO_MAS_WORK_DIR=/fsx/zzsamshi/a-evolve/evolution_workdir/w4_baseline \\
       NEMO_MAS_MEMORY_PATH=$NEMO_MAS_WORK_DIR/memory/records.jsonl \\
       /fsx/bnghe/miniconda3/bin/python \\
       /fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner/k8s/ablation_watcher.py'

Inspect with ``tmux attach -t ablation_watcher`` (Ctrl-b d to detach).
Stop with ``tmux kill-session -t ablation_watcher``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── config ───────────────────────────────────────────────────────────

WORK_DIR  = Path(os.environ["NEMO_MAS_WORK_DIR"])
KCTX      = os.environ.get(
    "KUBECTL_CTX",
    "arn:aws:eks:ap-southeast-3:801953956576:cluster/p5-llm",
)
BACKEND   = Path("/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner")
SUBMIT_SH = BACKEND / "k8s" / "submit.sh"
POLL_SEC  = int(os.environ.get("ABLATION_WATCHER_POLL_SEC", "30"))
EVAL_TP   = int(os.environ.get("ABLATION_WATCHER_EVAL_TP", "1"))

# Same eval-eligible nodes as eval_watcher; the scheduler queues overflow.
EVAL_NODES = [
    "ip-172-31-90-7.ap-southeast-3.compute.internal",
    "ip-172-31-95-1.ap-southeast-3.compute.internal",
]

CLI = [sys.executable, "-m", "agent_evolve.model.algorithms.nemo_mas.cli"]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── kubectl + ledger primitives ──────────────────────────────────────

def job_status(job_name: str) -> str:
    """Return 'succeeded' | 'failed' | 'running' | 'gone'.

    'gone' = no such Job (the controller may have GC'd it after TTL).
    """
    res = subprocess.run(
        ["kubectl", "--context", KCTX, "get", "job", job_name,
         "-o", "jsonpath={.status.succeeded}|{.status.failed}|{.status.active}"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        # likely "NotFound"; treat as gone.
        return "gone"
    succ, fail, active = (res.stdout.strip() + "||").split("|", 2)
    if succ.strip() == "1":
        return "succeeded"
    if fail and fail.strip() and fail.strip() != "0":
        return "failed"
    if active and active.strip() and active.strip() != "0":
        return "running"
    return "running"  # controller hasn't recorded counts yet


def append_record(role: str, kind: str, title: str, body: str,
                  refs: list[str], tags: list[str] | None = None) -> str | None:
    tmp = WORK_DIR / f".watcher_tmp_{int(time.time() * 1000)}.md"
    tmp.write_text(body)
    args = CLI + ["mem", "append", "--role", role, "--kind", kind,
                  "--title", title, "--body-file", str(tmp)]
    for r in refs:
        args += ["--ref", r]
    for t in (tags or []):
        args += ["--tag", t]
    res = subprocess.run(args, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if res.returncode != 0:
        log(f"  ! mem append failed ({kind}): "
            f"{(res.stderr or res.stdout).strip()}")
        return None
    last = res.stdout.strip().splitlines()[-1]
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        log(f"  ! mem append: unparseable stdout: {last!r}")
        return None
    rec = obj.get("record") or obj
    return rec.get("id")


def metric_read(ckpt_final_dir: Path) -> dict[str, Any]:
    """Read sidecar metric JSON via the nemo_mas CLI, returning {} on failure."""
    res = subprocess.run(
        CLI + ["metric", "read", "--ckpt", str(ckpt_final_dir)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return {}
    last = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {}


# ── arm helpers ──────────────────────────────────────────────────────

def pick_eval_node(used_nodes: list[str]) -> str:
    for n in EVAL_NODES:
        if n not in used_nodes:
            return n
    return EVAL_NODES[0]


def harvest_training_run(arm: dict[str, Any], marker: dict[str, Any],
                         arm_label: str) -> str | None:
    """Write a `training_run` record for a freshly-finished arm.

    Returns the record id, or None on append failure.
    """
    ctx = marker["context"]
    ckpt_final = Path(arm["ckpt_out"]) / "final"
    metric = metric_read(ckpt_final)
    primary_metric = (
        f"{metric.get('primary_metric_name','?')}={metric.get('primary_metric_value','?')}"
        if metric else "(metric.json missing — check the pod log)"
    )
    body = (
        f"recipe_path: {arm.get('recipe_path','?')}\n"
        f"data_path:   {arm['data_path']}\n"
        f"rows:        {arm.get('rows', '?')}\n"
        f"job_id:      {arm.get('job_name','?')}\n"
        f"final_ckpt:  {ckpt_final}\n"
        f"num_steps:   {ctx.get('num_steps','?')}\n"
        f"category:    {ctx.get('category','?')}\n"
        f"label:       {arm['label']}\n"
        f"status:      success\n"
        f"primary_metric: {primary_metric}\n"
        f"notes: auto-harvested by ablation_watcher (arm={arm_label})\n"
    )
    refs: list[str] = []
    # Refs — schema requires recipe_proposal + dataset_snapshot for
    # training_run. For ablation arms, the launch is lead-authorized so
    # neither is guaranteed. Best-effort: pull from marker.refs (which
    # holds distill_batch_id) and accept that the schema may reject.
    refs += marker.get("refs", [])
    rec_id = append_record(
        "trainer", "training_run",
        f"sft (ablation {arm['label']}): {arm.get('job_name','reused')}",
        body, refs, tags=[f"category:{ctx.get('category','?')}",
                          f"ablation_arm:{arm_label}"],
    )
    return rec_id


def submit_arm_eval(arm: dict[str, Any], marker: dict[str, Any],
                    arm_label: str, parent_training_run_id: str) -> dict | None:
    """Submit one eval Job for a freshly-finished arm. Drops the standard
    `ne-eval-…` marker so `trainer-collect-results` (or the Skill) can
    harvest later. Returns the eval-marker dict on success.
    """
    prefix = marker.get("run_name_prefix") or marker["context"].get(
        "category", "ablation"
    )
    eval_run_name = f"{prefix}-{arm_label}-eval"
    eval_job_name = f"ne-eval-{eval_run_name}"
    out_dir = WORK_DIR / "artifacts" / "eval" / f"{prefix}_{arm_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    used_nodes: list[str] = []
    if marker["context"].get("arm_a", {}).get("eval_node"):
        used_nodes.append(marker["context"]["arm_a"]["eval_node"])
    if marker["context"].get("arm_b", {}).get("eval_node"):
        used_nodes.append(marker["context"]["arm_b"]["eval_node"])
    eval_node = pick_eval_node(used_nodes)

    eval_marker = {
        "kind": "eval_report",
        "job_name": eval_job_name,
        "submitted_at": datetime.now(timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "node_pin": eval_node,
        "refs": [parent_training_run_id],
        "context": {
            "parent_id":   parent_training_run_id,
            "parent_kind": "training_run",
            "ckpt_path":   f"{arm['ckpt_out']}/final",
            "out_dir":     str(out_dir),
            "run_name":    eval_run_name,
            "split":       marker["context"].get(
                "expected_eval_split", "balanced_dev726",
            ),
            "tp":          EVAL_TP,
            "auto_dispatched": True,
            "ablation_marker": marker.get("run_name_prefix"),
            "arm": arm_label,
        },
    }
    eval_marker_path = WORK_DIR / ".pending_jobs" / f"{eval_job_name}.json"
    eval_marker_path.write_text(json.dumps(eval_marker, indent=2) + "\n")

    cmd = [str(SUBMIT_SH), "eval",
           "--adapter", f"{arm['ckpt_out']}/final",
           "--out",     str(out_dir),
           "--name",    eval_run_name,
           "--tp",      str(EVAL_TP),
           "--node",    eval_node]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"  ! submit.sh eval failed for {eval_run_name}: "
            f"{(res.stderr or res.stdout).strip()}")
        return None

    log(f"  ✓ {eval_job_name} dispatched on {eval_node}, "
        f"parent={parent_training_run_id}")
    eval_marker["context"]["eval_node"] = eval_node
    return eval_marker


def advance_to_phase2(marker_path: Path) -> bool:
    """Drive one ablation marker from awaiting_train → awaiting_evals.
    Returns True if the marker was advanced, False if it was left alone.
    """
    marker = json.loads(marker_path.read_text())
    ctx = marker.get("context", {})
    phase = ctx.get("phase", "awaiting_train")
    if phase != "awaiting_train":
        return False  # not our concern

    arm_a = ctx.get("arm_a", {})
    arm_b = ctx.get("arm_b", {})

    # Arm A status: cache-reused arms count as "succeeded".
    if arm_a.get("reused"):
        a_status = "succeeded"
    elif arm_a.get("job_name"):
        a_status = job_status(arm_a["job_name"])
    else:
        a_status = "gone"

    if arm_b.get("job_name"):
        b_status = job_status(arm_b["job_name"])
    else:
        b_status = "gone"

    name = marker_path.name
    if a_status == "running" or b_status == "running":
        return False
    if a_status == "failed" or b_status == "failed":
        log(f"  ! {name}: arm_a={a_status} arm_b={b_status} — "
            f"skipping; trainer-ablation-collect will write failed_attempt")
        return False
    if a_status == "gone" and not arm_a.get("reused"):
        log(f"  ! {name}: arm_a job gone with no record; deferring to LLM harvest")
        return False
    if b_status == "gone":
        log(f"  ! {name}: arm_b job gone with no record; deferring to LLM harvest")
        return False

    log(f"{name}: both train arms succeeded — advancing to phase 2")

    # Harvest training_runs for arms that actually trained.
    train_run_a = arm_a.get("training_run_id")  # set by launch on cache hit
    train_run_b = arm_b.get("training_run_id")
    if not train_run_a:
        train_run_a = harvest_training_run(arm_a, marker, "a")
        if not train_run_a:
            log(f"  ! {name}: failed to write training_run for arm_a; "
                f"leaving marker for LLM harvest")
            return False
    if not train_run_b:
        train_run_b = harvest_training_run(arm_b, marker, "b")
        if not train_run_b:
            log(f"  ! {name}: failed to write training_run for arm_b; "
                f"leaving marker for LLM harvest")
            return False
    arm_a["training_run_id"] = train_run_a
    arm_b["training_run_id"] = train_run_b

    # Submit eval Jobs for arms without an existing eval_report_id.
    if not arm_a.get("eval_report_id"):
        em = submit_arm_eval(arm_a, marker, "a", train_run_a)
        if em is None:
            log(f"  ! {name}: arm_a eval submit failed; aborting phase advance")
            return False
        arm_a["eval_job_name"] = em["job_name"]
        arm_a["eval_node"]     = em["context"]["eval_node"]
        arm_a["eval_out_dir"]  = em["context"]["out_dir"]
    if not arm_b.get("eval_report_id"):
        em = submit_arm_eval(arm_b, marker, "b", train_run_b)
        if em is None:
            log(f"  ! {name}: arm_b eval submit failed; aborting phase advance")
            return False
        arm_b["eval_job_name"] = em["job_name"]
        arm_b["eval_node"]     = em["context"]["eval_node"]
        arm_b["eval_out_dir"]  = em["context"]["out_dir"]

    # Flip phase + persist.
    ctx["phase"]  = "awaiting_evals"
    ctx["arm_a"] = arm_a
    ctx["arm_b"] = arm_b
    ctx["phase_advanced_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    marker["context"] = ctx
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    log(f"  ✓ {name}: advanced to awaiting_evals "
        f"(train_a={train_run_a}, train_b={train_run_b})")
    return True


# ── loop ─────────────────────────────────────────────────────────────

def loop() -> int:
    if not WORK_DIR.is_dir():
        log(f"work dir not found: {WORK_DIR}")
        return 2
    pending = WORK_DIR / ".pending_jobs"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / "done").mkdir(exist_ok=True)
    log(f"watching {pending}/ablation-*.json  poll={POLL_SEC}s")
    while True:
        scanned = 0
        advanced = 0
        ready_for_phase2 = 0
        for marker_path in sorted(pending.glob("ablation-*.json")):
            scanned += 1
            try:
                marker = json.loads(marker_path.read_text())
            except json.JSONDecodeError:
                log(f"  ! {marker_path.name}: unparseable JSON; skipping")
                continue
            phase = marker.get("context", {}).get("phase", "awaiting_train")
            if phase == "awaiting_evals":
                ready_for_phase2 += 1
                continue
            try:
                if advance_to_phase2(marker_path):
                    advanced += 1
            except Exception as exc:
                log(f"  ! {marker_path.name}: advance failed: {exc!r}")
        if scanned or advanced:
            log(f"tick: scanned={scanned} advanced={advanced} "
                f"awaiting_evals_ready_for_LLM={ready_for_phase2}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        sys.exit(loop() or 0)
    except KeyboardInterrupt:
        log("stopped")
        sys.exit(0)
