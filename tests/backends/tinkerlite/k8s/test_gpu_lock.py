"""Unit tests for ``gpu_lock`` — flock correctness + stale PID cleanup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_evolve.backends.tinkerlite.k8s.gpu_lock import (
    acquire_gpus,
    live_locked_gpus,
)


def test_acquire_and_release_basic(tmp_path: Path) -> None:
    lease = acquire_gpus(3, pool=(0, 1, 2, 3), lock_dir=tmp_path, trial_id="t1")
    assert lease is not None
    assert lease.gpu_ids == [0, 1, 2]
    assert live_locked_gpus(tmp_path) == {0, 1, 2}

    lease.release()
    assert live_locked_gpus(tmp_path) == set()


def test_acquire_insufficient_returns_none(tmp_path: Path) -> None:
    # Take 3 out of pool of 4.
    first = acquire_gpus(3, pool=(0, 1, 2, 3), lock_dir=tmp_path, trial_id="t1")
    assert first is not None

    # Second caller asks for 2 — only 1 slot free.
    second = acquire_gpus(2, pool=(0, 1, 2, 3), lock_dir=tmp_path, trial_id="t2")
    assert second is None
    # First still holds its 3.
    assert live_locked_gpus(tmp_path) == {0, 1, 2}

    first.release()


def test_stale_lock_cleanup_via_dead_pid(tmp_path: Path) -> None:
    """A lock file written by a dead pid should be auto-reclaimed."""
    # Pretend-stale lock file: dead pid, flock released.
    stale = tmp_path / "gpu_0.lock"
    stale.write_text("pid=1\ntrial=ghost\n")  # pid=1 is init; alive but not ours.
    # Force a genuinely-dead pid by writing an impossibly high pid.
    stale.write_text("pid=99999999\ntrial=ghost\n")

    # Probe triggers sweep.
    locked = live_locked_gpus(tmp_path)
    assert 0 not in locked  # swept out

    # Now acquiring gpu 0 should succeed.
    lease = acquire_gpus(1, pool=(0, 1), lock_dir=tmp_path, trial_id="live")
    assert lease is not None
    assert lease.gpu_ids == [0]
    lease.release()


def test_context_manager_releases_on_exit(tmp_path: Path) -> None:
    with acquire_gpus(2, pool=(0, 1, 2), lock_dir=tmp_path, trial_id="ctx") as lease:
        assert lease is not None
        assert set(lease.gpu_ids) == {0, 1}
        assert live_locked_gpus(tmp_path) == {0, 1}
    # After context exit, locks released.
    assert live_locked_gpus(tmp_path) == set()


def test_double_release_is_safe(tmp_path: Path) -> None:
    lease = acquire_gpus(1, pool=(0, 1), lock_dir=tmp_path, trial_id="t1")
    assert lease is not None
    lease.release()
    lease.release()  # must not raise
    assert live_locked_gpus(tmp_path) == set()


def test_acquire_rolls_back_partial_allocation(tmp_path: Path) -> None:
    """If we ask for more than available, no partial locks should remain."""
    # Occupy GPU 1.
    blocker = acquire_gpus(1, pool=(1,), lock_dir=tmp_path, trial_id="blocker")
    assert blocker is not None

    # Ask for 2 from {0, 1, 2}. GPU 0 is free, 1 is taken, 2 is free — total free 2,
    # requested 2. With the skip-locked path, we should end up with {0, 2}.
    lease = acquire_gpus(2, pool=(0, 1, 2), lock_dir=tmp_path, trial_id="t1")
    assert lease is not None
    assert set(lease.gpu_ids) == {0, 2}

    lease.release()
    blocker.release()


def test_acquire_more_than_pool_returns_none(tmp_path: Path) -> None:
    lease = acquire_gpus(5, pool=(0, 1), lock_dir=tmp_path, trial_id="t1")
    assert lease is None
    assert live_locked_gpus(tmp_path) == set()
