#!/usr/bin/env python3
"""eval_watcher — auto-dispatch eval Jobs for new SFT checkpoints.

Polls ``<work_dir>/artifacts/sft/*/step_<N>/`` every POLL_SEC seconds.
For each newly-saved checkpoint that hasn't been evaluated yet, it:

  1. Reads the training marker at ``.pending_jobs/ne-train-<run_short>.json``
     (or ``.pending_jobs/done/<...>``) to recover the recipe + dataset ids
     and the node the training pod is pinned to.
  2. Writes a ``profile_run`` ledger record for the checkpoint (so the
     eventual ``eval_report`` has a valid parent).
  3. Writes the eval marker at ``.pending_jobs/ne-eval-<run_short>-step<N>.json``
     **before** invoking submit.sh, so a watcher restart mid-launch can't
     double-submit.
  4. Calls ``submit.sh eval`` to dispatch the k8s Job.

Skip rules — a checkpoint is left alone if any of:

  * ``artifacts/eval/<run_short>_step<N>/metrics.json`` already exists.
  * ``.pending_jobs/ne-eval-<run_short>-step<N>.json`` exists (in flight).
  * ``.pending_jobs/done/ne-eval-<...>.json`` exists (already harvested).

Run via tmux::

    tmux new -d -s eval_watcher \\
      'NEMO_MAS_WORK_DIR=/fsx/zzsamshi/a-evolve/evolution_workdir/w4_baseline \\
       NEMO_MAS_MEMORY_PATH=$NEMO_MAS_WORK_DIR/memory/records.jsonl \\
       /fsx/bnghe/miniconda3/bin/python \\
       /fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner/k8s/eval_watcher.py'

Inspect with ``tmux attach -t eval_watcher`` (Ctrl-b d to detach without killing).
Stop with ``tmux kill-session -t eval_watcher``.

The harvester (``trainer-collect-results``) reads the markers this script
writes — wave-1, wave-2, and any future runs all flow through the same path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── config ───────────────────────────────────────────────────────────

WORK_DIR  = Path(os.environ["NEMO_MAS_WORK_DIR"])
KCTX      = os.environ.get(
    "KUBECTL_CTX",
    "arn:aws:eks:ap-southeast-3:801953956576:cluster/p5-llm",
)
BACKEND   = Path("/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner")
SUBMIT_SH = BACKEND / "k8s" / "submit.sh"
POLL_SEC  = int(os.environ.get("EVAL_WATCHER_POLL_SEC", "30"))
EVAL_TP   = int(os.environ.get("EVAL_WATCHER_TP", "1"))

# Eval-eligible nodes. We pick the one NOT running the run's training pod.
EVAL_NODES = [
    "ip-172-31-90-7.ap-southeast-3.compute.internal",
    "ip-172-31-95-1.ap-southeast-3.compute.internal",
]

CLI = [sys.executable, "-m", "agent_evolve.model.algorithms.nemo_mas.cli"]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── discovery ────────────────────────────────────────────────────────

def discover_checkpoints():
    """Yield ``(run_short, step, abs_ckpt_path)`` for every step dir on disk."""
    for ckpt in sorted(WORK_DIR.glob("artifacts/sft/*/step_*")):
        if not ckpt.is_dir():
            continue
        if not (ckpt / "adapter_config.json").exists():
            continue  # save still in progress
        if not (ckpt / "adapter_model.safetensors").exists():
            continue  # weights file not finalized
        run_short = ckpt.parent.name
        step_str  = ckpt.name.removeprefix("step_")
        if not step_str.isdigit():
            continue
        yield run_short, int(step_str), ckpt


def already_handled(run_short: str, step: int) -> str | None:
    out_dir = WORK_DIR / "artifacts" / "eval" / f"{run_short}_step{step}"
    if (out_dir / "metrics.json").exists():
        return "metrics.json present"
    pending = WORK_DIR / ".pending_jobs" / f"ne-eval-{run_short}-step{step}.json"
    if pending.exists():
        return "marker pending"
    done = WORK_DIR / ".pending_jobs" / "done" / f"ne-eval-{run_short}-step{step}.json"
    if done.exists():
        return "marker harvested"
    return None


def load_train_marker(run_short: str) -> dict | None:
    for cand in [
        WORK_DIR / ".pending_jobs" / f"ne-train-{run_short}.json",
        WORK_DIR / ".pending_jobs" / "done" / f"ne-train-{run_short}.json",
    ]:
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def pick_eval_node(train_node: str) -> str:
    """Prefer the non-training node so eval doesn't fight training for slots.

    The eval Job YAML uses ``nodeSelector:`` (soft pin) — the scheduler
    will queue overflow pods as ``Pending`` if the chosen node fills up,
    rather than kubelet-rejecting. So the watcher just picks a preferred
    host; it doesn't need to gate on free-GPU count.
    """
    for n in EVAL_NODES:
        if n != train_node:
            return n
    return EVAL_NODES[0]  # both nodes are training; let scheduler queue


# ── ledger + dispatch ────────────────────────────────────────────────

def append_record(role: str, kind: str, title: str, body: str,
                  refs: list[str]) -> str | None:
    tmp = WORK_DIR / f".watcher_tmp_{int(time.time() * 1000)}.md"
    tmp.write_text(body)
    args = CLI + ["mem", "append", "--role", role, "--kind", kind,
                  "--title", title, "--body-file", str(tmp)]
    for r in refs:
        args += ["--ref", r]
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


def submit_eval(run_short: str, step: int, ckpt: Path,
                train_marker: dict) -> None:
    eval_name = f"{run_short}-step{step}"
    job_name  = f"ne-eval-{eval_name}"
    out_dir   = WORK_DIR / "artifacts" / "eval" / f"{run_short}_step{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx        = train_marker["context"]
    recipe_id  = ctx["recipe_id"]
    dataset_id = ctx["dataset_id"]
    train_node = train_marker.get("node_pin", "")

    body = (
        f"recipe_id: {recipe_id}\n"
        f"recipe_path: {ctx['recipe_path']}\n"
        f"data_path: {ctx['data_path']}\n"
        f"ckpt_path: {ckpt}\n"
        f"step: {step}\n"
        f"training_job_id: {train_marker['job_name']}\n"
        f"diff_summary: {ctx.get('diff_summary', '')}\n"
        f"notes: auto-dispatched by eval_watcher\n"
    )
    profile_id = append_record(
        "trainer", "profile_run",
        f"profile: {run_short} step_{step} (auto)",
        body, [recipe_id, dataset_id],
    )
    if profile_id is None:
        log(f"  ! skipping {eval_name}: profile_run append failed")
        return

    eval_node = pick_eval_node(train_node)
    marker = {
        "kind": "eval_report",
        "job_name": job_name,
        "submitted_at": datetime.now(timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "node_pin": eval_node,
        "refs": [profile_id],
        "context": {
            "parent_id":   profile_id,
            "parent_kind": "profile_run",
            "ckpt_path":   str(ckpt),
            "out_dir":     str(out_dir),
            "run_name":    eval_name,
            "split":       "balanced_dev726",
            "tp":          EVAL_TP,
            "auto_dispatched": True,
        },
    }
    marker_path = WORK_DIR / ".pending_jobs" / f"{job_name}.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")

    cmd = [str(SUBMIT_SH), "eval",
           "--adapter", str(ckpt),
           "--out",     str(out_dir),
           "--name",    eval_name,
           "--tp",      str(EVAL_TP)]
    if eval_node:
        cmd += ["--node", eval_node]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"  ! submit.sh failed for {eval_name}: "
            f"{(res.stderr or res.stdout).strip()}")
        # marker stays — trainer-collect-results will see no Job and
        # write a failed_attempt on its next pass.
        return

    log(f"  ✓ {job_name} dispatched on "
        f"{eval_node or '<scheduler>'}, profile_run={profile_id}")


# ── loop ─────────────────────────────────────────────────────────────

def loop() -> int:
    if not WORK_DIR.is_dir():
        log(f"work dir not found: {WORK_DIR}")
        return 2
    pending = WORK_DIR / ".pending_jobs"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / "done").mkdir(exist_ok=True)
    log(f"watching {WORK_DIR}/artifacts/sft/  poll={POLL_SEC}s  tp={EVAL_TP}")
    while True:
        scanned = 0
        dispatched = 0
        for run_short, step, ckpt in discover_checkpoints():
            scanned += 1
            why = already_handled(run_short, step)
            if why:
                continue
            tm = load_train_marker(run_short)
            if tm is None:
                log(f"  ? no training marker for {run_short}; "
                    f"leaving {ckpt.name} alone")
                continue
            log(f"new ckpt: {run_short}/step_{step}")
            submit_eval(run_short, step, ckpt, tm)
            dispatched += 1
        if dispatched:
            log(f"tick done: scanned {scanned}, dispatched {dispatched}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        sys.exit(loop() or 0)
    except KeyboardInterrupt:
        log("stopped")
        sys.exit(0)
