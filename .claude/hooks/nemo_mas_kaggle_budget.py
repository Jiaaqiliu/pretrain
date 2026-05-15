#!/usr/bin/env python3
"""PreToolUse hook — enforce the per-run Kaggle submission budget.

Wired via ``.claude/settings.json``::

    "hooks": {
      "PreToolUse": [
        {"matcher": "mcp__nemo_mas__kaggle_submit",
         "type": "command",
         "command": "/fsx/zzsamshi/a-evolve/.claude/hooks/nemo_mas_kaggle_budget.py"}
      ]
    }

We count ``kaggle_submission_result`` records in the active ledger and
reject the call once ``max_kaggle_submits_per_run`` has been reached.
The cap lives in ``<NEMO_MAS_WORK_DIR>/meta.json`` so each run can set
its own (default: 1). Override at the session level via
``NEMO_MAS_KAGGLE_MAX_PER_RUN`` env var.

Exit 0 = allow. Exit 2 = block with stderr shown to the teammate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_MAX = 1


def main() -> int:
    try:
        from agent_evolve.model.algorithms.nemo_mas.agent_teams import (
            count_records_of_kind,
            read_meta,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[nemo_mas kaggle hook] import failed: {e}", file=sys.stderr)
        return 0

    meta = read_meta()
    max_submits = int(
        os.environ.get("NEMO_MAS_KAGGLE_MAX_PER_RUN")
        or meta.get("max_kaggle_submits_per_run", _DEFAULT_MAX)
    )
    done = count_records_of_kind("kaggle_submission_result")
    if done >= max_submits:
        print(
            f"[nemo_mas] Kaggle budget exhausted: {done}/{max_submits} "
            f"submits already made this run. Post `verdict=ready_to_sign` "
            f"on `cp_submission_ready` without submitting — the human "
            f"can push manually via the Kaggle CLI if needed. "
            f"Raise `max_kaggle_submits_per_run` in "
            f"`<work_dir>/meta.json` to lift the cap.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
