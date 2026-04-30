"""Unit tests for ``ElasticScheduler.probe_capacity`` fan-out sizing.

Formula (see scheduler.py::probe_capacity):
  recommended = k8s_run_now + (k8s_queue_budget if k8s.can_queue else 0) + local_run_now
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from agent_evolve.backends.tinkerlite.elastic.compute_target import (
    CapacityReport,
    TargetHandle,
)
from agent_evolve.backends.tinkerlite.elastic.scheduler import (
    ElasticScheduler,
    FanoutCapacity,
)


@dataclass
class FakeTarget:
    name: str
    priority: int
    report: CapacityReport

    def capacity_probe(self, required_gpus: int) -> CapacityReport:
        return self.report

    # Remaining methods unused by probe_capacity — stubbed to satisfy Protocol.
    def submit(self, *a, **kw): raise NotImplementedError
    def poll(self, handle): return "pending"
    def wait(self, handle, timeout=None): raise NotImplementedError
    def wait_with_pending_timeout(self, handle, pending_timeout): raise NotImplementedError
    def cancel(self, handle): pass


def test_idle_cluster_recommends_run_now_plus_queue() -> None:
    """K8s has 16 free GPUs (can run 2 at world_size=8), can queue, plus
    queue budget of 4. Local has 8 free (1 run_now). Expected: 2+4+1 = 7."""
    k8s = FakeTarget("k8s", 0, CapacityReport(True, True, 16, "cluster idle"))
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "local free"))
    sched = ElasticScheduler([k8s, local], k8s_queue_budget=4)

    cap = sched.probe_capacity(world_size=8)
    assert isinstance(cap, FanoutCapacity)
    assert cap.recommended == 7
    assert cap.breakdown == {
        "k8s_run_now": 2,
        "local_run_now": 1,
        "k8s_queue_budget": 4,
    }


def test_saturated_cluster_still_allows_queue_plus_local() -> None:
    """Cluster full (0 free), but can queue → queue_budget contributes.
    Local has 1 slot. Expected: 0+4+1 = 5."""
    k8s = FakeTarget("k8s", 0, CapacityReport(False, True, 0, "all busy"))
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "local free"))
    sched = ElasticScheduler([k8s, local], k8s_queue_budget=4)

    cap = sched.probe_capacity(world_size=8)
    assert cap.recommended == 5
    assert cap.breakdown["k8s_queue_budget"] == 4


def test_cluster_unavailable_falls_back_to_local_only() -> None:
    """No matching nodes → can_queue=False → queue budget zeroed out.
    Only local's 1 slot remains."""
    k8s = FakeTarget("k8s", 0, CapacityReport(False, False, 0, "no H200 nodes"))
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "local free"))
    sched = ElasticScheduler([k8s, local], k8s_queue_budget=4)

    cap = sched.probe_capacity(world_size=8)
    assert cap.recommended == 1
    assert cap.breakdown["k8s_queue_budget"] == 0


def test_nothing_available_but_k8s_can_queue_returns_one() -> None:
    """Degenerate: local is out, k8s is saturated with 0 GPU — but
    queue_budget=0. The fallback path still allows 1 submission so
    the caller isn't stuck."""
    k8s = FakeTarget("k8s", 0, CapacityReport(False, True, 0, "busy"))
    local = FakeTarget("local", 10, CapacityReport(False, False, 0, "all locked"))
    sched = ElasticScheduler([k8s, local], k8s_queue_budget=0)

    cap = sched.probe_capacity(world_size=8)
    # k8s_run_now=0 + queue_budget=0 + local_run_now=0 = 0 → fallback raises to 1.
    assert cap.recommended == 1


def test_fully_unavailable_returns_zero() -> None:
    """Neither k8s nor local can accept anything — recommended=0 signals
    the driver should back off entirely."""
    k8s = FakeTarget("k8s", 0, CapacityReport(False, False, 0, "no nodes"))
    local = FakeTarget("local", 10, CapacityReport(False, False, 0, "locked"))
    sched = ElasticScheduler([k8s, local], k8s_queue_budget=4)

    cap = sched.probe_capacity(world_size=8)
    assert cap.recommended == 0


def test_queue_budget_configurable() -> None:
    """k8s_queue_budget=0 means "never hold pending pods" — only dispatch
    what can run immediately. This is the friendly-to-other-tenants mode."""
    k8s = FakeTarget("k8s", 0, CapacityReport(False, True, 0, "busy"))
    local = FakeTarget("local", 10, CapacityReport(True, False, 8, "free"))
    sched = ElasticScheduler([k8s, local], k8s_queue_budget=0)

    cap = sched.probe_capacity(world_size=8)
    assert cap.recommended == 1   # local run_now only
    assert cap.breakdown["k8s_queue_budget"] == 0


def test_world_size_affects_run_now_count() -> None:
    """32 free GPUs @ world_size=4 → 8 concurrent trials from k8s alone."""
    k8s = FakeTarget("k8s", 0, CapacityReport(True, True, 32, "lots of space"))
    local = FakeTarget("local", 10, CapacityReport(False, False, 0, "local gone"))
    sched = ElasticScheduler([k8s, local], k8s_queue_budget=2)

    cap = sched.probe_capacity(world_size=4)
    assert cap.breakdown["k8s_run_now"] == 8
    assert cap.recommended == 8 + 2  # + queue budget


def test_backend_exposes_probe(tmp_path: Path, monkeypatch) -> None:
    """The backend surface exposes probe_fanout_capacity; driver code
    depends on this name."""
    from agent_evolve.backends.tinkerlite.elastic import K8sTinkerLiteBackend

    # Stub out the kubernetes-bound K8sComputeTarget construction so we
    # can instantiate the backend without kubeconfig.
    import agent_evolve.backends.tinkerlite.elastic.backend as backend_mod

    class _StubK8sTarget:
        name = "k8s"
        priority = 0
        def capacity_probe(self, n):
            return CapacityReport(False, True, 0, "stub")
        def submit(self, *a, **kw): raise NotImplementedError
        def poll(self, h): return "pending"
        def wait(self, h, timeout=None): raise NotImplementedError
        def wait_with_pending_timeout(self, h, pt): raise NotImplementedError
        def cancel(self, h): pass

    monkeypatch.setattr(backend_mod, "K8sComputeTarget", lambda **kw: _StubK8sTarget())

    backend = K8sTinkerLiteBackend(
        local_enabled=True,
        local_gpu_pool=(0, 1, 2, 3, 4, 5, 6, 7),
        local_lock_dir=tmp_path,
        k8s_queue_budget=3,
    )
    cap = backend.probe_fanout_capacity(world_size=8)
    assert isinstance(cap, FanoutCapacity)
    # k8s stub: run_now=0, can_queue=True → queue_budget=3.
    # local: (tmp_path empty) → 1 slot run_now.
    # Total expected: 0 + 3 + 1 = 4 (assuming nvidia-smi absent in tests).
    assert cap.recommended >= 3
