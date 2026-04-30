"""``ComputeTarget`` — abstraction over "where a DDP stage runs".

Both the local subprocess path and the k8s Job path implement this Protocol,
so the ElasticScheduler can treat them uniformly. The execution contract is:

  1. Caller writes a ``.ddp_config.json`` on shared storage (FSx).
  2. ``submit(cfg_path, world_size, log_dir) -> TargetHandle`` —  non-blocking.
  3. ``capacity_probe(required_gpus)`` — non-blocking, returns availability.
  4. ``poll(handle)``  — non-blocking status peek.
  5. ``wait(handle, timeout)`` / ``wait_with_pending_timeout(handle, pending_timeout)``
     — blocking; reads ``.ddp_result.json`` on success.
  6. ``cancel(handle)`` — best-effort termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class CapacityReport:
    """Point-in-time snapshot of a target's ability to accept work.

    ``can_run_now``   — target has enough free GPUs to start *now*.
    ``can_queue``     — target can hold the job and start later (k8s only).
                        Local never queues; set to False.
    ``available_gpus``— free capacity (best-effort; None if unknown).
    ``reason``        — human-readable diagnostic for logs.
    """
    can_run_now: bool
    can_queue: bool
    available_gpus: int | None
    reason: str


@dataclass
class TargetHandle:
    """Opaque handle returned by ``submit``.

    Fields are target-specific; the scheduler only reads ``cfg_path`` /
    ``result_path`` / ``target_name`` generically.
    """
    target_name: str
    cfg_path: Path
    result_path: Path
    # Target-specific payload (k8s: job_name+namespace; local: Popen object).
    inner: Any


class CapacityExhausted(RuntimeError):
    """Raised when no ComputeTarget can accept the stage and queueing is
    either disabled or has timed out."""


class PendingTimeout(TimeoutError):
    """Raised by ``wait_with_pending_timeout`` when a job stayed pending
    past the threshold. Caller is expected to cancel and fall back."""


@runtime_checkable
class ComputeTarget(Protocol):
    name: str
    priority: int   # lower = preferred. K8s=0, Local=10 by convention.

    def capacity_probe(self, required_gpus: int) -> CapacityReport: ...

    def submit(
        self,
        cfg_path: Path,
        world_size: int,
        log_dir: Path,
        *,
        stage_label: str = "stage",
    ) -> TargetHandle: ...

    def poll(self, handle: TargetHandle) -> Literal["pending", "running", "succeeded", "failed"]: ...

    def wait(self, handle: TargetHandle, timeout: float | None = None) -> dict: ...

    def wait_with_pending_timeout(
        self,
        handle: TargetHandle,
        pending_timeout: float,
    ) -> dict:
        """Raise ``PendingTimeout`` if the job stayed pending past the
        threshold; otherwise block until the job finishes."""
        ...

    def cancel(self, handle: TargetHandle) -> None: ...


__all__ = [
    "CapacityReport",
    "TargetHandle",
    "ComputeTarget",
    "CapacityExhausted",
    "PendingTimeout",
]
