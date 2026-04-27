"""Eval worker — executes an :class:`EvalPlan`.

Smoke mode writes a deterministic ``metrics.json`` + ``predictions.jsonl``
so the benchmark parser has something to read. Real inference is deferred to
a PR that wires up a real sampling stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import EvalPlan


def run_eval_plan(
    plan: EvalPlan,
    *,
    smoke: bool = True,
) -> Path:
    out = Path(plan.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if smoke:
        _write_smoke_artifacts(plan, out)
        return out

    # Real inference hook — future PR plugs in a sampling client here.
    raise NotImplementedError("Non-smoke eval path is not available in this PR")


def _write_smoke_artifacts(plan: EvalPlan, out: Path) -> None:
    # Read holdout rows if they exist so the primary metric is at least
    # correlated with dataset content.
    rows = _load_holdout_rows(plan)
    if rows:
        correct = sum(1 for row in rows if row.get("is_correct"))
        primary = correct / max(1, len(rows))
    else:
        primary = 0.0

    # Primary metric name must align with the benchmark's ``primary_metric``.
    metric_key = "local_holdout_pass_at_1"
    metrics: dict[str, Any] = {
        metric_key: primary,
        "primary": primary,
        "format_error_rate": 0.0,
        "avg_output_tokens": 0.0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with open(out / "predictions.jsonl", "w") as f:
        for row in rows or []:
            f.write(json.dumps(row) + "\n")


def _load_holdout_rows(plan: EvalPlan) -> list[dict]:
    root = Path(plan.config_path).parent  # eval/ directory
    candidates = [
        root / "local_holdout_small.jsonl",
        root / "local_holdout.jsonl",
    ]
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
