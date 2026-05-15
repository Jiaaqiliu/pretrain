#!/usr/bin/env python3
"""PreToolUse hook for the ``Agent`` tool — log nemo_mas teammate spawns.

Wired via ``.claude/settings.json`` as a PreToolUse hook with
matcher=``Agent``. Whenever the orchestrator (or any other caller)
spawns a subagent with ``subagent_type`` starting with ``nemo_mas_``,
this hook writes a ``task_assignment`` record into the active ledger
capturing what was assigned + to whom + when.

Contract (per https://code.claude.com/docs/en/hooks):
  - stdin: JSON with ``tool_name`` and ``tool_input``.
  - exit 0: allow the spawn (this hook always allows; it only logs).
  - exit 2: block the spawn (we never block here).

This hook is purely additive: it logs and returns. It never mutates
the spawn's prompt, never blocks, never raises.

Behavior on error:
  - Missing env vars (no active ledger) → log to stderr, allow spawn.
  - Non-nemo_mas subagent type → silently allow (don't pollute the ledger).
  - mem-append CLI failure → log to stderr with the CLI's error, allow spawn.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _allow_and_exit(reason: str = "") -> None:
    if reason:
        print(f"[nemo_mas_agent_spawn] {reason}", file=sys.stderr)
    sys.exit(0)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        _allow_and_exit(f"could not parse hook stdin as JSON: {e}")

    if payload.get("tool_name") != "Agent":
        _allow_and_exit()

    tool_input = payload.get("tool_input") or {}
    subagent_type = str(tool_input.get("subagent_type") or "")
    if not subagent_type.startswith("nemo_mas_"):
        _allow_and_exit()  # not our concern

    role = subagent_type[len("nemo_mas_"):]  # planner | trainer | data_worker
    prompt = tool_input.get("prompt") or ""
    description = tool_input.get("description") or ""

    if not os.environ.get("NEMO_MAS_MEMORY_PATH") and not os.environ.get("NEMO_MAS_WORK_DIR"):
        _allow_and_exit("no active ledger (NEMO_MAS_MEMORY_PATH / NEMO_MAS_WORK_DIR unset); skipping log")

    # Body written to a temp file so the prompt may contain backticks /
    # quotes / heredocs without breaking shell escaping.
    body_lines = [
        f"target_role: {role}",
        f"subagent_type: {subagent_type}",
        f"description: {description}",
        "",
        "## Prompt as delivered",
        "",
        prompt,
    ]
    body = "\n".join(body_lines)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".md",
        prefix="nemo_mas_task_assignment_",
        delete=False,
    ) as tf:
        tf.write(body)
        body_path = tf.name

    cli = [
        sys.executable, "-m",
        "agent_evolve.model.algorithms.nemo_mas.cli", "mem", "append",
        "--role", "main",
        "--kind", "task_assignment",
        "--title", f"spawn {role}: {description[:140]}",
        "--body-file", body_path,
        "--tag", f"subagent:{role}",
        "--tag", "channel:spawn_log",
    ]

    try:
        out = subprocess.run(
            cli, capture_output=True, text=True, timeout=15,
            cwd=str(_REPO_ROOT),
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _allow_and_exit(f"mem append timed out / failed: {e}")
    finally:
        try:
            os.unlink(body_path)
        except OSError:
            pass

    if out.returncode != 0:
        # CLI failed (usually a schema-validation error or missing env var).
        # We still allow the spawn; the missing log is a soft loss.
        msg = (out.stderr or out.stdout or "").strip()
        _allow_and_exit(f"mem append rejected: {msg}")

    # Success — print a one-line marker to stderr (visible to the operator
    # but doesn't interfere with the tool call). The CLI may emit one or
    # more JSON lines on stdout; the record append is always the LAST one
    # whose ``ok`` is true and that carries an ``id``.
    rec_id = "?"
    for line in reversed(out.stdout.strip().splitlines()):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not (isinstance(obj, dict) and obj.get("ok")):
            continue
        # CLI nests the record under ``record`` (mem append shape) but
        # other subcommands sometimes inline ``id`` at the top level.
        rec = obj.get("record") if isinstance(obj.get("record"), dict) else obj
        if rec.get("id"):
            rec_id = rec["id"]
            break
    print(f"[nemo_mas_agent_spawn] logged spawn of {subagent_type} as {rec_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
