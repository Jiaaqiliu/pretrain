"""TrainingObserver — persists per-cycle/per-trial artifacts.

The observer is deliberately dumb: it only serializes to disk. Reward,
incumbent selection, and validity live elsewhere (MCGS + benchmark).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrainingObserver:
    def __init__(self, evolution_dir: str | Path):
        self.root = Path(evolution_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "reports").mkdir(exist_ok=True)
        (self.root / "observations").mkdir(exist_ok=True)

    def record_trial(self, trial_result: Any) -> Path:
        obs_path = self.root / "observations" / f"{trial_result.node_id}.json"
        with open(obs_path, "w") as f:
            json.dump(_coerce(trial_result), f, indent=2, default=str)
        return obs_path

    def record_cycle(self, report: Any) -> Path:
        cycle_num = getattr(report, "cycle", 0)
        path = self.root / "reports" / f"cycle_{cycle_num:04d}.json"
        with open(path, "w") as f:
            json.dump(_coerce(report), f, indent=2, default=str)
        return path

    def append_run(self, memory_root: Path, row: dict) -> None:
        memory_root.mkdir(parents=True, exist_ok=True)
        runs = memory_root / "runs.jsonl"
        row = dict(row)
        row.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        with open(runs, "a") as f:
            f.write(json.dumps(row) + "\n")


def _coerce(value: Any) -> Any:
    if is_dataclass(value):
        return _coerce(asdict(value))
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    return value
