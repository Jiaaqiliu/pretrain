"""``ElasticScheduler`` — route a DDP stage to the best available target.

Policy is explicit in the class, not spread across targets:

1. Ask every target (in priority order) whether it can run *now*. First
   yes wins → submit + wait.
2. If nobody can run now, try to queue on the primary (k8s). If primary
   doesn't support queueing (``can_queue=False``, e.g. cluster has zero
   matching nodes), skip to step 3.
3. Try the remaining targets in priority order for either now-or-queue.
4. If all fail, raise ``CapacityExhausted``. Caller decides whether to
   surface as a trial failure or retry later.

The scheduler also exposes a non-blocking ``submit_async`` for callers
that want explicit fan-out (e.g. a parallel LR sweep driver), returning
a ``StageHandle`` that wraps target + target-handle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .compute_target import (
    CapacityExhausted,
    CapacityReport,
    ComputeTarget,
    PendingTimeout,
    TargetHandle,
)

logger = logging.getLogger(__name__)


@dataclass
class StageHandle:
    """Handle returned by the scheduler's async submit.

    The scheduler's ``wait_any`` only needs ``target`` + ``target_handle``.
    """
    target: ComputeTarget
    target_handle: TargetHandle
    stage_label: str = ""
    submitted_at: float = field(default_factory=time.time)


@dataclass
class FanoutCapacity:
    """Answer to "how many ``world_size``-sized trials can I dispatch now?"

    ``recommended`` is the scheduler's single-number suggestion —
    ``k8s_run_now + min(k8s_queue_budget, queue_slots_soft_cap) + local_run_now``.
    ``breakdown`` exposes the per-target contributions so callers that want
    to impose their own policy (e.g. "never use local") can.
    """
    recommended: int
    breakdown: dict[str, int]
    reason: str


class ElasticScheduler:
    def __init__(
        self,
        targets: Sequence[ComputeTarget],
        *,
        queue_timeout_secs: float = 600.0,
        stage_hard_timeout_secs: float | None = None,
        k8s_queue_budget: int = 4,
    ):
        if not targets:
            raise ValueError("ElasticScheduler requires at least one target")
        # Ensure priority order; callers can pass arbitrary order.
        self.targets: list[ComputeTarget] = sorted(targets, key=lambda t: t.priority)
        self.queue_timeout_secs = float(queue_timeout_secs)
        self.stage_hard_timeout_secs = stage_hard_timeout_secs
        # Max number of jobs we'll voluntarily leave in k8s Pending on top
        # of what the cluster can run now. Prevents a driver from dumping
        # 100 LR trials into the shared queue and angering other tenants.
        self.k8s_queue_budget = int(k8s_queue_budget)

    # ── Blocking API ────────────────────────────────────────────────

    def run_stage(
        self,
        cfg_path: Path,
        world_size: int,
        log_dir: Path,
        *,
        stage_label: str = "stage",
        mode: str = "ddp",
    ) -> dict:
        """Submit + wait for a stage. Returns the parsed result JSON.

        ``mode`` is forwarded to each target's ``submit`` and selects the
        pod entrypoint: ``"ddp"`` (default) runs torchrun + ddp_worker;
        ``"eval"`` runs eval_worker directly (no torchrun).

        Policy is strictly priority-first: the primary target (k8s by
        convention) gets both the run-now and the queue opportunity before
        we consult any fallback. Only after the primary has exhausted both
        paths do we try secondary targets. This matches the "k8s-priority"
        contract — we prefer a queued k8s job over running on local.
        """
        reports: list[tuple[ComputeTarget, CapacityReport]] = [
            (t, t.capacity_probe(world_size)) for t in self.targets
        ]
        for t, r in reports:
            logger.info("[elastic] %s: %s", t.name, r.reason)

        # Step 1: primary target (priority-first). Try run-now, then queue.
        primary, primary_report = reports[0]
        if primary_report.can_run_now:
            logger.info(
                "[elastic] %s ready (avail=%s); submitting",
                primary.name, primary_report.available_gpus,
            )
            handle = primary.submit(
                cfg_path, world_size, log_dir,
                stage_label=stage_label, mode=mode,
            )
            return primary.wait(handle, timeout=self.stage_hard_timeout_secs)

        if primary_report.can_queue:
            logger.info(
                "[elastic] %s has no immediate capacity; queueing up to %.0fs",
                primary.name, self.queue_timeout_secs,
            )
            handle = primary.submit(
                cfg_path, world_size, log_dir,
                stage_label=stage_label, mode=mode,
            )
            try:
                return primary.wait_with_pending_timeout(
                    handle, pending_timeout=self.queue_timeout_secs,
                )
            except PendingTimeout as exc:
                logger.warning("[elastic] primary queue timed out: %s; falling back", exc)
                primary.cancel(handle)

        # Step 2: fall back through remaining targets in priority order.
        for t, r in reports[1:]:
            if r.can_run_now:
                logger.info("[elastic] falling back to %s (run-now)", t.name)
                handle = t.submit(
                    cfg_path, world_size, log_dir,
                    stage_label=stage_label, mode=mode,
                )
                return t.wait(handle, timeout=self.stage_hard_timeout_secs)
            if r.can_queue:
                logger.info("[elastic] falling back to %s (queued)", t.name)
                handle = t.submit(
                    cfg_path, world_size, log_dir,
                    stage_label=stage_label, mode=mode,
                )
                return t.wait_with_pending_timeout(
                    handle, pending_timeout=self.queue_timeout_secs,
                )

        raise CapacityExhausted(
            "No compute target can run or queue this stage. "
            + "; ".join(f"{t.name}: {r.reason}" for t, r in reports)
        )

    # ── Non-blocking API (parallel fan-out) ─────────────────────────

    def submit_async(
        self,
        cfg_path: Path,
        world_size: int,
        log_dir: Path,
        *,
        stage_label: str = "stage",
    ) -> StageHandle:
        """Pick a target (priority-first: primary run-now or queue, then
        fallback run-now or queue) and return without blocking. Caller
        uses ``wait_any`` to collect.
        """
        reports = [(t, t.capacity_probe(world_size)) for t in self.targets]
        # Primary first — accept either run-now or queue.
        primary, primary_report = reports[0]
        if primary_report.can_run_now or primary_report.can_queue:
            handle = primary.submit(cfg_path, world_size, log_dir, stage_label=stage_label)
            return StageHandle(target=primary, target_handle=handle, stage_label=stage_label)
        # Fallbacks in priority order — accept either run-now or queue.
        for t, r in reports[1:]:
            if r.can_run_now or r.can_queue:
                handle = t.submit(cfg_path, world_size, log_dir, stage_label=stage_label)
                return StageHandle(target=t, target_handle=handle, stage_label=stage_label)
        raise CapacityExhausted(
            "No target can accept an async submission right now. "
            + "; ".join(f"{t.name}: {r.reason}" for t, r in reports)
        )

    def wait_any(
        self,
        handles: list[StageHandle],
        *,
        poll_interval_secs: float = 5.0,
    ) -> tuple[StageHandle, dict]:
        """Block until any handle finishes; return (handle, result)."""
        import json
        if not handles:
            raise ValueError("wait_any called with no handles")
        while True:
            for sh in handles:
                phase = sh.target.poll(sh.target_handle)
                if phase == "succeeded":
                    if not sh.target_handle.result_path.is_file():
                        raise RuntimeError(
                            f"{sh.target.name} reported success but no result at "
                            f"{sh.target_handle.result_path}"
                        )
                    return sh, json.loads(sh.target_handle.result_path.read_text())
                if phase == "failed":
                    raise RuntimeError(
                        f"stage '{sh.stage_label}' on target {sh.target.name} failed"
                    )
            time.sleep(poll_interval_secs)

    def cancel(self, handle: StageHandle) -> None:
        handle.target.cancel(handle.target_handle)

    # ── Capacity advice for fan-out drivers ─────────────────────────

    def probe_capacity(self, world_size: int) -> FanoutCapacity:
        """Estimate how many ``world_size``-sized trials can dispatch now.

        Formula:
          recommended = k8s_run_now + min(k8s_queue_budget, 1 if can_queue else 0)
                        * k8s_queue_budget + local_run_now

        The intent is: fill cluster capacity first, optionally hold a
        bounded queue, and add local as a cushion. Callers may override
        by reading ``breakdown``.
        """
        breakdown: dict[str, int] = {}
        reasons: list[str] = []

        for t in self.targets:
            r = t.capacity_probe(world_size)
            avail = int(r.available_gpus or 0)
            run_now = avail // world_size if world_size > 0 else 0
            breakdown[f"{t.name}_run_now"] = run_now
            reasons.append(f"{t.name}: run_now={run_now} ({r.reason})")

        # Queue budget only applies to the primary (priority 0) and only
        # if it can_queue. Local never queues.
        primary = self.targets[0]
        primary_report = primary.capacity_probe(world_size)
        queue_slots = self.k8s_queue_budget if primary_report.can_queue else 0
        breakdown[f"{primary.name}_queue_budget"] = queue_slots

        recommended = sum(breakdown.values())
        if recommended < 1:
            # Degenerate case: nothing free right now. Still let the caller
            # submit 1 — it'll either fail-fast or join the queue if any
            # target can hold work.
            if primary_report.can_queue:
                recommended = 1
                reasons.append("fallback: allow 1 submission to join primary queue")
            else:
                recommended = 0

        return FanoutCapacity(
            recommended=recommended,
            breakdown=breakdown,
            reason="; ".join(reasons),
        )


__all__ = ["ElasticScheduler", "StageHandle", "FanoutCapacity"]
