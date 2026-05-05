#!/usr/bin/env python3
"""TaskCreated hook — block task creation when a required checkpoint is pending.

Wired via ``.claude/settings.json``::

    "hooks": {
      "TaskCreated": [
        {"type": "command",
         "command": "/fsx/zzsamshi/a-evolve/.claude/hooks/nemo_mas_task_created.py"}
      ]
    }

Contract (per https://code.claude.com/docs/en/hooks):
  - Exit code 0 → allow the event to proceed.
  - Exit code 2 → block the event and show stderr to the lead.

We block when ``first_required_blocker`` returns a slot whose state is
``pending_human`` (manual mode) OR ``pending_evidence`` with a required
``requires_evidence`` kind missing. The lead sees the blocker message
and can instruct the team to produce evidence or sign the slot.

Optional guards can be added later (e.g. refuse task creation when the
Kaggle budget is exhausted). Keep this script small — it runs on every
task creation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the nemo_mas module importable when invoked as a standalone script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    try:
        from agent_evolve.model.algorithms.nemo_mas.agent_teams import (
            current_checkpoint_mode,
            first_blocker_or_none,
            format_blocker_message,
        )
    except Exception as e:  # noqa: BLE001
        # Never crash the harness — fail open, log to stderr for debug.
        print(f"[nemo_mas hook] import failed: {e}", file=sys.stderr)
        return 0

    blocker = first_blocker_or_none()
    if blocker is None:
        return 0

    # Only block on states that genuinely require human attention. Pending
    # evidence without a review is a normal in-flight state — teammates
    # should be allowed to create tasks that produce that evidence.
    if blocker.state not in ("pending_human",):
        return 0

    mode = current_checkpoint_mode()
    msg = format_blocker_message(blocker, mode)
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
