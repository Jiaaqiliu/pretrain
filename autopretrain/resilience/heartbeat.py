"""Heartbeat reader: monitors training liveness via FSx file protocol.

Training scripts write heartbeat JSON files to a shared directory.
The agent reads these files to detect:
- Silent hangs (process alive but not making progress)
- Crashes (heartbeat file stops updating)
- Startup failures (no heartbeat file created)

Protocol:
- Training writes: /fsx/.../heartbeat/{trial_id}.json
- Agent reads every poll_interval
- Stale threshold: 2 min (process may be doing checkpoint)
- Dead threshold: 6 min (process is definitely stuck)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from autopretrain.core.types import Heartbeat

logger = logging.getLogger(__name__)


class FSxHeartbeatReader:
    """Reads heartbeat files from a shared filesystem."""

    def __init__(self, heartbeat_dir: str | Path) -> None:
        self._dir = Path(heartbeat_dir)

    async def read_heartbeat(self, trial_id: str) -> Heartbeat | None:
        """Read the latest heartbeat for a trial.

        Returns None if no heartbeat file exists (trial hasn't started writing yet).
        """
        path = self._dir / f"{trial_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return Heartbeat(
                trial_id=data.get("trial_id", trial_id),
                timestamp=data["timestamp"],
                step=data.get("step", 0),
                loss=data.get("loss"),
                grad_norm=data.get("grad_norm"),
                throughput_tps=data.get("throughput_tps"),
                gpu_memory_pct=data.get("gpu_memory_pct"),
                status=data.get("status", "unknown"),
            )
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning("Error reading heartbeat for %s: %s", trial_id, e)
            return None

    async def list_active_heartbeats(self) -> list[Heartbeat]:
        """List all heartbeats that are not dead (< 6 min old)."""
        if not self._dir.exists():
            return []

        active: list[Heartbeat] = []
        for path in self._dir.glob("*.json"):
            trial_id = path.stem
            hb = await self.read_heartbeat(trial_id)
            if hb and not hb.is_dead:
                active.append(hb)

        return active

    async def clear_heartbeat(self, trial_id: str) -> None:
        """Remove a heartbeat file (e.g., after trial completes)."""
        path = self._dir / f"{trial_id}.json"
        if path.exists():
            path.unlink()
