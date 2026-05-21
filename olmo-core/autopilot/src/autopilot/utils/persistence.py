"""State persistence for long-running agent operations."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from autopilot.utils.logging import get_logger

log = get_logger("persistence")


@dataclass
class ExperimentRecord:
    experiment_id: str
    name: str
    config: Dict[str, Any]
    status: str  # "pending", "running", "completed", "failed", "stopped"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DecisionRecord:
    decision_id: str
    experiment_id: str
    decision_type: str  # "early_stop", "adjust_lr", "adjust_mixture", "rollback", "launch"
    reasoning: str
    action: Dict[str, Any]
    outcome: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StateStore:
    """File-based persistent state store for the agent.

    Stores experiment records, decision history, and agent state in a
    JSON-backed store for durability across restarts.
    """

    def __init__(self, store_dir: str | Path):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._experiments_file = self._dir / "experiments.json"
        self._decisions_file = self._dir / "decisions.json"
        self._state_file = self._dir / "agent_state.json"
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._decisions: List[DecisionRecord] = []
        self._state: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._experiments_file.exists():
            data = json.loads(self._experiments_file.read_text())
            self._experiments = {
                k: ExperimentRecord.from_dict(v) for k, v in data.items()
            }
        if self._decisions_file.exists():
            data = json.loads(self._decisions_file.read_text())
            self._decisions = [DecisionRecord.from_dict(d) for d in data]
        if self._state_file.exists():
            self._state = json.loads(self._state_file.read_text())

    def _save_experiments(self) -> None:
        data = {k: v.to_dict() for k, v in self._experiments.items()}
        self._experiments_file.write_text(json.dumps(data, indent=2))

    def _save_decisions(self) -> None:
        data = [d.to_dict() for d in self._decisions]
        self._decisions_file.write_text(json.dumps(data, indent=2))

    def _save_state(self) -> None:
        self._state_file.write_text(json.dumps(self._state, indent=2))

    def save_experiment(self, record: ExperimentRecord) -> None:
        record.updated_at = time.time()
        self._experiments[record.experiment_id] = record
        self._save_experiments()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        return self._experiments.get(experiment_id)

    def list_experiments(
        self, status: Optional[str] = None
    ) -> List[ExperimentRecord]:
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)

    def save_decision(self, record: DecisionRecord) -> None:
        self._decisions.append(record)
        self._save_decisions()
        if record.experiment_id in self._experiments:
            exp = self._experiments[record.experiment_id]
            exp.decisions.append(record.to_dict())
            self.save_experiment(exp)

    def get_decisions(
        self, experiment_id: Optional[str] = None, limit: int = 50
    ) -> List[DecisionRecord]:
        decisions = self._decisions
        if experiment_id:
            decisions = [d for d in decisions if d.experiment_id == experiment_id]
        return sorted(decisions, key=lambda d: d.timestamp, reverse=True)[:limit]

    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value
        self._save_state()

    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)
