"""Unit tests for ``LocalComputeTarget.capacity_probe``.

We stub out ``nvidia-smi`` (it may or may not exist in CI) and exercise
the interaction with the gpu_lock state.
"""

from __future__ import annotations

from pathlib import Path

from agent_evolve.backends.tinkerlite.k8s import local_target as lt_mod
from agent_evolve.backends.tinkerlite.k8s.gpu_lock import acquire_gpus
from agent_evolve.backends.tinkerlite.k8s.local_target import LocalComputeTarget


def test_probe_no_smi_no_locks(tmp_path: Path, monkeypatch) -> None:
    # Simulate nvidia-smi missing entirely.
    monkeypatch.setattr(lt_mod, "_nvidia_smi_free_gpus", lambda *a, **kw: set())
    target = LocalComputeTarget(gpu_pool=(0, 1, 2, 3), lock_dir=tmp_path)
    r = target.capacity_probe(required_gpus=2)
    # With no smi data and no locks, we trust locks-only: 4 free, 2 required.
    assert r.can_run_now
    assert r.can_queue is False
    assert r.available_gpus == 4


def test_probe_respects_live_locks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lt_mod, "_nvidia_smi_free_gpus", lambda *a, **kw: set())
    # Acquire 3 locks in our own process — they're "live".
    lease = acquire_gpus(3, pool=(0, 1, 2, 3), lock_dir=tmp_path, trial_id="pytest")
    assert lease is not None
    try:
        target = LocalComputeTarget(gpu_pool=(0, 1, 2, 3), lock_dir=tmp_path)
        r = target.capacity_probe(required_gpus=2)
        assert r.available_gpus == 1
        assert not r.can_run_now
    finally:
        lease.release()


def test_probe_smi_intersects_with_locks(tmp_path: Path, monkeypatch) -> None:
    # smi reports only {2, 3} as free (mem pressure on 0, 1).
    monkeypatch.setattr(lt_mod, "_nvidia_smi_free_gpus", lambda *a, **kw: {2, 3})
    target = LocalComputeTarget(gpu_pool=(0, 1, 2, 3), lock_dir=tmp_path)
    r = target.capacity_probe(required_gpus=2)
    assert r.available_gpus == 2
    assert r.can_run_now
    assert not r.can_queue


def test_probe_can_queue_is_always_false(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lt_mod, "_nvidia_smi_free_gpus", lambda *a, **kw: set())
    target = LocalComputeTarget(gpu_pool=(0, 1), lock_dir=tmp_path)
    r = target.capacity_probe(required_gpus=8)
    # Asking for 8 on a 2-slot pool: can_run_now=False, can_queue=False.
    assert not r.can_run_now
    assert not r.can_queue
