"""Unit tests for ``ElasticScheduler`` routing logic.

Uses a ``FakeComputeTarget`` that records submissions and lets us drive
capacity reports + wait outcomes. No k8s, no subprocess, no GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent_evolve.backends.tinkerlite.elastic.compute_target import (
    CapacityExhausted,
    CapacityReport,
    PendingTimeout,
    TargetHandle,
)
from agent_evolve.backends.tinkerlite.elastic.scheduler import ElasticScheduler


@dataclass
class FakeTarget:
    name: str
    priority: int
    report: CapacityReport
    wait_result: dict = field(default_factory=lambda: {"ok": True})
    wait_raises: Exception | None = None
    submitted: list[TargetHandle] = field(default_factory=list)
    canceled: list[TargetHandle] = field(default_factory=list)

    def capacity_probe(self, required_gpus: int) -> CapacityReport:
        return self.report

    def submit(self, cfg_path, world_size, log_dir, *, stage_label: str = "stage",
               mode: str = "ddp") -> TargetHandle:
        h = TargetHandle(
            target_name=self.name,
            cfg_path=Path(cfg_path),
            result_path=Path(cfg_path).with_suffix(".result.json"),
            inner={"label": stage_label, "mode": mode},
        )
        self.submitted.append(h)
        return h

    def poll(self, handle): return "succeeded"
    def wait(self, handle, timeout=None):
        if self.wait_raises:
            raise self.wait_raises
        return self.wait_result

    def wait_with_pending_timeout(self, handle, pending_timeout):
        if self.wait_raises:
            raise self.wait_raises
        return self.wait_result

    def cancel(self, handle):
        self.canceled.append(handle)


def _cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text('{"out_result_path": "whatever"}')
    return p


def test_runs_on_first_available_target(tmp_path: Path) -> None:
    k8s = FakeTarget("k8s", 0, CapacityReport(True, True, 8, "ok"), {"from": "k8s"})
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "ok"), {"from": "local"})
    sched = ElasticScheduler([local, k8s])  # passed out of order on purpose

    result = sched.run_stage(_cfg(tmp_path), 8, tmp_path, stage_label="t")
    # K8s has priority 0 → preferred.
    assert result == {"from": "k8s"}
    assert len(k8s.submitted) == 1
    assert len(local.submitted) == 0


def test_queues_on_k8s_when_no_immediate_capacity(tmp_path: Path) -> None:
    k8s = FakeTarget("k8s", 0, CapacityReport(False, True, 0, "busy"), {"from": "k8s-queued"})
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "free"))
    sched = ElasticScheduler([k8s, local], queue_timeout_secs=60)

    # No target can run now — but k8s can queue, and its wait_with_pending_timeout
    # succeeds (doesn't raise).
    result = sched.run_stage(_cfg(tmp_path), 8, tmp_path)
    assert result == {"from": "k8s-queued"}
    assert len(k8s.submitted) == 1
    assert len(local.submitted) == 0


def test_falls_back_to_local_on_pending_timeout(tmp_path: Path) -> None:
    k8s = FakeTarget("k8s", 0, CapacityReport(False, True, 0, "busy"),
                     wait_raises=PendingTimeout("timeout"))
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "free"),
                       wait_result={"from": "local"})
    sched = ElasticScheduler([k8s, local], queue_timeout_secs=1)

    result = sched.run_stage(_cfg(tmp_path), 8, tmp_path)
    assert result == {"from": "local"}
    # k8s was submitted then canceled.
    assert len(k8s.submitted) == 1
    assert len(k8s.canceled) == 1
    assert len(local.submitted) == 1


def test_zero_cluster_capacity_skips_straight_to_local(tmp_path: Path) -> None:
    # can_queue=False (no matching nodes) — scheduler must NOT wait on k8s.
    k8s = FakeTarget("k8s", 0, CapacityReport(False, False, 0, "no H200 nodes"))
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "free"),
                       wait_result={"from": "local"})
    sched = ElasticScheduler([k8s, local])

    result = sched.run_stage(_cfg(tmp_path), 8, tmp_path)
    assert result == {"from": "local"}
    assert len(k8s.submitted) == 0
    assert len(local.submitted) == 1


def test_capacity_exhausted_when_nobody_can_accept(tmp_path: Path) -> None:
    k8s = FakeTarget("k8s", 0, CapacityReport(False, False, 0, "no nodes"))
    local = FakeTarget("local", 10, CapacityReport(False, False, 0, "all locked"))
    sched = ElasticScheduler([k8s, local])

    with pytest.raises(CapacityExhausted, match="no nodes"):
        sched.run_stage(_cfg(tmp_path), 8, tmp_path)


def test_submit_async_picks_runnable_target(tmp_path: Path) -> None:
    k8s = FakeTarget("k8s", 0, CapacityReport(False, True, 0, "busy"))
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "free"))
    sched = ElasticScheduler([k8s, local])

    sh = sched.submit_async(_cfg(tmp_path), 8, tmp_path, stage_label="s1")
    # No target can run_now — but k8s can queue, which submit_async prefers.
    assert sh.target.name == "k8s"


def test_priority_order_independent_of_input_order(tmp_path: Path) -> None:
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "free"),
                       wait_result={"from": "local"})
    k8s = FakeTarget("k8s", 0, CapacityReport(True, True, 8, "free"),
                     wait_result={"from": "k8s"})
    # Pass local first.
    sched = ElasticScheduler([local, k8s])
    assert sched.run_stage(_cfg(tmp_path), 8, tmp_path) == {"from": "k8s"}
