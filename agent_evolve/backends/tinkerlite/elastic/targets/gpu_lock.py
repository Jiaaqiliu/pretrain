"""File-lock based GPU allocation for the local compute target.

Multi-process local scheduling without a daemon: each live trial holds a
``flock(LOCK_EX|LOCK_NB)`` on ``/fsx/.ae_locks/gpu_{i}.lock`` for every GPU
it owns. A crashed process releases the OS flock automatically; as a
second line of defense we also parse the ``pid`` written inside the lock
file and treat the lock as stale if ``os.kill(pid, 0)`` reports the
process is gone.

The lock directory lives on FSx so peers across hosts / terminals see the
same state. Chmod to 0777 on creation — this is a multi-user shared FSx.
"""

from __future__ import annotations

import errno
import fcntl
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_LOCK_DIR = Path("/fsx/.ae_locks")


@dataclass
class _HeldLock:
    gpu_id: int
    path: Path
    fd: int


class GpuLockError(RuntimeError):
    pass


class GpuLease:
    """Holds a set of GPU locks for the lifetime of a trial.

    Use as a context manager or call ``release()`` explicitly. Safe to call
    ``release()`` multiple times. On interpreter crash the OS drops the
    ``flock``; the pid-stale sweep handles any leftover lock files.
    """

    def __init__(self, held: list[_HeldLock]):
        self._held = held
        self.gpu_ids: list[int] = sorted(h.gpu_id for h in held)

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        while self._held:
            h = self._held.pop()
            try:
                fcntl.flock(h.fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(h.fd)
            except OSError:
                pass
            # Best-effort unlink; on multi-host FSx another process may
            # race us — ignore ENOENT.
            try:
                h.path.unlink()
            except OSError:
                pass


def _ensure_lock_dir(lock_dir: Path) -> None:
    if not lock_dir.exists():
        lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(lock_dir, 0o777)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM  # alive, but owned by another user
    return True


def _sweep_stale(lock_dir: Path) -> None:
    """Remove ``gpu_*.lock`` files whose pid is dead. Best-effort."""
    if not lock_dir.exists():
        return
    for p in lock_dir.glob("gpu_*.lock"):
        try:
            raw = p.read_text().strip()
        except OSError:
            continue
        m = re.search(r"pid=(\d+)", raw)
        if not m:
            # Malformed / empty lock file — try to reclaim it.
            _try_reclaim(p)
            continue
        if not _pid_alive(int(m.group(1))):
            _try_reclaim(p)


def _try_reclaim(path: Path) -> None:
    """Attempt to take-and-release the flock to confirm the holder is dead,
    then unlink. If flock fails with EAGAIN, something is actually holding
    it — leave the file alone."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    try:
        path.unlink()
    except OSError:
        pass


def live_locked_gpus(lock_dir: Path = _DEFAULT_LOCK_DIR) -> set[int]:
    """Return GPU ids currently held by live processes."""
    _ensure_lock_dir(lock_dir)
    _sweep_stale(lock_dir)
    held: set[int] = set()
    for p in lock_dir.glob("gpu_*.lock"):
        m = re.match(r"gpu_(\d+)\.lock$", p.name)
        if not m:
            continue
        # If we can't flock it (EAGAIN), someone live holds it.
        try:
            fd = os.open(p, os.O_RDWR)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Got it — holder is gone; release and skip.
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                held.add(int(m.group(1)))
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
    return held


def acquire_gpus(
    count: int,
    *,
    pool: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    lock_dir: Path = _DEFAULT_LOCK_DIR,
    trial_id: str = "",
) -> GpuLease | None:
    """Try to grab ``count`` GPUs from ``pool``. Returns None on failure
    (not enough free GPUs). Caller should release via ``GpuLease.release()``
    or ``with ... as lease``.

    Retries on ECHILD/race — if a pool member is locked between probe and
    acquire, we skip it and try the next. Only fails if the whole pool is
    exhausted.
    """
    _ensure_lock_dir(lock_dir)
    _sweep_stale(lock_dir)

    held: list[_HeldLock] = []
    pid = os.getpid()
    stamp = f"pid={pid}\ntrial={trial_id}\nts={time.time()}\n"

    for gpu_id in pool:
        if len(held) == count:
            break
        path = lock_dir / f"gpu_{gpu_id}.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        # Record holder for diagnostics (best-effort; truncate+write).
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, stamp.encode())
            os.fsync(fd)
        except OSError:
            pass
        held.append(_HeldLock(gpu_id=gpu_id, path=path, fd=fd))

    if len(held) < count:
        # Roll back any partial allocation.
        GpuLease(held).release()
        return None
    return GpuLease(held)


__all__ = [
    "GpuLease",
    "GpuLockError",
    "acquire_gpus",
    "live_locked_gpus",
]
