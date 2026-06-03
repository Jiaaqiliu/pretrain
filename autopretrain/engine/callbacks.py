"""Custom OLMo-core callbacks injected into training scripts.

These callbacks provide the bridge between the training process and the
external orchestrator agent:

1. HeartbeatCallback: Writes liveness signal to FSx every N steps
2. MetricReporterCallback: Writes detailed metrics for the agent to read
3. AgentControlCallback: Reads control signals from agent (pause/stop/adjust)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict

log = logging.getLogger(__name__)


# NOTE: These callbacks are GENERATED into training scripts, not imported at runtime.
# The code below serves as the canonical implementation that gets embedded.


HEARTBEAT_CALLBACK_CODE = '''
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Any

from olmo_core.train.callbacks.callback import Callback


@dataclass
class HeartbeatCallback(Callback):
    """Writes heartbeat file every N steps for external liveness monitoring.

    The orchestrator agent reads this file to detect silent hangs.
    If no update for 6 minutes, the agent assumes the process is dead.
    """

    priority: ClassVar[int] = -10  # Run after other callbacks
    heartbeat_dir: str = ""
    trial_id: str = ""
    interval: int = 10  # Write every N steps

    _last_loss: float = 0.0
    _last_grad_norm: float = 0.0
    _step_start_time: float = 0.0
    _tokens_per_step: int = 0

    def pre_train(self):
        Path(self.heartbeat_dir).mkdir(parents=True, exist_ok=True)
        self._step_start_time = time.time()

    def post_step(self):
        if self.step % self.interval != 0:
            return

        # Collect metrics from trainer
        metrics = {}
        if hasattr(self.trainer, '_latest_metrics'):
            metrics = self.trainer._latest_metrics

        loss = metrics.get("train/ce_loss", self._last_loss)
        grad_norm = metrics.get("optim/grad_norm", self._last_grad_norm)
        self._last_loss = loss
        self._last_grad_norm = grad_norm

        # Estimate throughput
        elapsed = time.time() - self._step_start_time
        throughput = None
        if elapsed > 0 and self._tokens_per_step > 0:
            throughput = self._tokens_per_step / elapsed
        self._step_start_time = time.time()

        # Write heartbeat
        heartbeat = {
            "trial_id": self.trial_id,
            "timestamp": time.time(),
            "step": self.step,
            "loss": float(loss) if loss else None,
            "grad_norm": float(grad_norm) if grad_norm else None,
            "throughput_tps": throughput,
            "gpu_memory_pct": None,  # Filled by GPUMemoryMonitor if available
            "status": "training",
        }

        path = Path(self.heartbeat_dir) / f"{self.trial_id}.json"
        try:
            path.write_text(json.dumps(heartbeat))
        except Exception as e:
            pass  # Non-fatal: don't crash training for heartbeat failure

    def post_train(self):
        """Write final heartbeat marking completion."""
        heartbeat = {
            "trial_id": self.trial_id,
            "timestamp": time.time(),
            "step": self.step,
            "loss": float(self._last_loss) if self._last_loss else None,
            "status": "completed",
        }
        path = Path(self.heartbeat_dir) / f"{self.trial_id}.json"
        try:
            path.write_text(json.dumps(heartbeat))
        except Exception:
            pass

    def on_error(self, exc: BaseException):
        """Write error heartbeat so agent knows immediately."""
        heartbeat = {
            "trial_id": self.trial_id,
            "timestamp": time.time(),
            "step": self.step,
            "status": "error",
            "error": str(exc)[:200],
        }
        path = Path(self.heartbeat_dir) / f"{self.trial_id}.json"
        try:
            path.write_text(json.dumps(heartbeat))
        except Exception:
            pass
'''


METRIC_REPORTER_CALLBACK_CODE = '''
@dataclass
class MetricReporterCallback(Callback):
    """Writes detailed metrics to a JSONL file for the agent to consume.

    More detailed than heartbeat — includes all tracked metrics.
    Written less frequently (every metrics_collect_interval steps).
    """

    priority: ClassVar[int] = -11
    metrics_dir: str = ""
    trial_id: str = ""

    def pre_train(self):
        Path(self.metrics_dir).mkdir(parents=True, exist_ok=True)

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        """Called right before metrics are logged — capture all of them."""
        if not metrics:
            return

        record = {
            "step": step,
            "timestamp": time.time(),
            "trial_id": self.trial_id,
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }

        path = Path(self.metrics_dir) / f"{self.trial_id}.jsonl"
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\\n")
        except Exception:
            pass
'''


def get_callback_code(trial_id: str, heartbeat_dir: str, metrics_dir: str) -> str:
    """Get the full callback code to inject into a training script."""
    return f"""
# === HoloPretrain Callbacks (auto-injected) ===
{HEARTBEAT_CALLBACK_CODE}

{METRIC_REPORTER_CALLBACK_CODE}

HEARTBEAT_DIR = "{heartbeat_dir}"
METRICS_DIR = "{metrics_dir}"
TRIAL_ID = "{trial_id}"
# === End HoloPretrain Callbacks ===
"""


def get_callback_registration(trial_id: str, heartbeat_dir: str, metrics_dir: str) -> str:
    """Get the trainer callback registration code."""
    return f"""
    .with_callback("heartbeat", HeartbeatCallback(
        heartbeat_dir="{heartbeat_dir}",
        trial_id="{trial_id}",
        interval=10,
    ))
    .with_callback("metric_reporter", MetricReporterCallback(
        metrics_dir="{metrics_dir}",
        trial_id="{trial_id}",
    ))"""
