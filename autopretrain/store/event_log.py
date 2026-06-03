"""JSONL event log for complete audit trail.

Every state transition, every decision, every action is logged here.
This enables:
- Post-mortem debugging
- Learning from past failures
- Accountability (what did the agent decide and why)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from autopretrain.core.types import JobEvent

logger = logging.getLogger(__name__)


class EventLog:
    """Append-only JSONL event log."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: JobEvent) -> None:
        """Append an event to the log file."""
        record = {
            "timestamp": event.timestamp,
            "trial_id": event.trial_id,
            "event_type": event.event_type,
            "from_state": event.from_state,
            "to_state": event.to_state,
            "details": event.details,
            "decision_reason": event.decision_reason,
            "evidence": event.evidence,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def query(
        self,
        trial_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[JobEvent]:
        """Query events from the log."""
        if not self._path.exists():
            return []

        events: list[JobEvent] = []
        with open(self._path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                if trial_id and record.get("trial_id") != trial_id:
                    continue
                if event_type and record.get("event_type") != event_type:
                    continue

                events.append(JobEvent(
                    timestamp=record["timestamp"],
                    trial_id=record["trial_id"],
                    event_type=record["event_type"],
                    from_state=record.get("from_state", ""),
                    to_state=record.get("to_state", ""),
                    details=record.get("details", {}),
                    decision_reason=record.get("decision_reason", ""),
                    evidence=record.get("evidence", []),
                ))

                if len(events) >= limit:
                    break

        return events

    def tail(self, n: int = 20) -> list[JobEvent]:
        """Get the last N events."""
        if not self._path.exists():
            return []

        lines = self._path.read_text().strip().split("\n")
        events: list[JobEvent] = []
        for line in lines[-n:]:
            if not line.strip():
                continue
            record = json.loads(line)
            events.append(JobEvent(
                timestamp=record["timestamp"],
                trial_id=record["trial_id"],
                event_type=record["event_type"],
                from_state=record.get("from_state", ""),
                to_state=record.get("to_state", ""),
                details=record.get("details", {}),
            ))
        return events
