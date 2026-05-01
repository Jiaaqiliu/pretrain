"""``LocalComputeTarget`` — run a DDP stage as a torchrun subprocess.

Semantically equivalent to ``single_node.ddp_launcher._spawn_torchrun`` but
wrapped in the ComputeTarget Protocol: non-blocking ``submit`` returns a
handle whose ``poll``/``wait`` consult the subprocess and the
``.ddp_result.json`` file.

GPU allocation is tracked via ``gpu_lock`` flocks so that multiple parent
processes (e.g. parallel LR sweep drivers) coexist safely on one node.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..compute_target import CapacityReport, PendingTimeout, TargetHandle
from .gpu_lock import GpuLease, acquire_gpus, live_locked_gpus


@dataclass
class _LocalInner:
    proc: subprocess.Popen
    lease: GpuLease
    gpu_ids: list[int]
    started_at: float


def _nvidia_smi_free_gpus(mem_free_min_mib: int = 40_000) -> set[int]:
    """Best-effort: GPUs with plenty of free memory. Returns an empty set
    if nvidia-smi is missing or misbehaves, so caller should combine with
    lock-file state.

    ``mem_free_min_mib`` defaults to 40 GiB — on H200 (80 GB) this filters
    out any GPU with a non-trivial resident model. Lower for smaller cards.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return set()
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            timeout=5.0,
        ).decode()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return set()
    free: set[int] = set()
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            mem_free = int(parts[1])
        except ValueError:
            continue
        if mem_free >= mem_free_min_mib:
            free.add(idx)
    return free


class LocalComputeTarget:
    name = "local"
    priority = 10  # after k8s

    def __init__(
        self,
        *,
        gpu_pool: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
        lock_dir: Path = Path("/fsx/.ae_locks"),
        smi_mem_free_min_mib: int = 40_000,
        ae_root: Path | None = None,
    ):
        self.gpu_pool = tuple(gpu_pool)
        self.lock_dir = Path(lock_dir)
        self.smi_mem_free_min_mib = smi_mem_free_min_mib
        # Points at the repo root so the subprocess can PYTHONPATH into it.
        self.ae_root = Path(ae_root) if ae_root else Path(__file__).resolve().parents[5]

    # ── Capacity ────────────────────────────────────────────────────

    def capacity_probe(self, required_gpus: int) -> CapacityReport:
        locked = live_locked_gpus(self.lock_dir)
        pool = set(self.gpu_pool)
        smi_free = _nvidia_smi_free_gpus(self.smi_mem_free_min_mib)
        # If nvidia-smi isn't available, trust the lock file exclusively.
        if smi_free:
            free = pool - locked - (pool - smi_free)
        else:
            free = pool - locked
        avail = len(free)
        return CapacityReport(
            can_run_now=avail >= required_gpus,
            can_queue=False,
            available_gpus=avail,
            reason=(
                f"local pool={sorted(pool)} locked={sorted(locked)} "
                f"smi_free={sorted(smi_free) if smi_free else 'n/a'} "
                f"=> free={sorted(free)}"
            ),
        )

    # ── Submission ──────────────────────────────────────────────────

    def submit(
        self,
        cfg_path: Path,
        world_size: int,
        log_dir: Path,
        *,
        stage_label: str = "stage",
    ) -> TargetHandle:
        cfg_path = Path(cfg_path)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        lease = acquire_gpus(
            world_size,
            pool=self.gpu_pool,
            lock_dir=self.lock_dir,
            trial_id=cfg_path.parent.name,
        )
        if lease is None:
            raise RuntimeError(
                f"local target cannot acquire {world_size} GPUs from "
                f"pool {self.gpu_pool}"
            )

        # Read the result path out of the cfg JSON so poll() knows where
        # to look for success.
        import json
        cfg = json.loads(cfg_path.read_text())
        result_path = Path(cfg["out_result_path"])

        env = os.environ.copy()
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("WANDB_DISABLED", "true")
        env["PYTHONPATH"] = str(self.ae_root) + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # Bind the subprocess to the leased GPUs.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in lease.gpu_ids)

        cmd = [
            sys.executable,
            "-m", "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--master_addr=127.0.0.1",
            f"--master_port={29500 + (hash(str(cfg_path)) % 1000)}",
            "-m", "agent_evolve.model.runners.ddp_worker",
            "--config", str(cfg_path),
        ]
        log_path = log_dir / f"{stage_label}.local.log"
        log_fp = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)

        return TargetHandle(
            target_name=self.name,
            cfg_path=cfg_path,
            result_path=result_path,
            inner=_LocalInner(
                proc=proc, lease=lease, gpu_ids=lease.gpu_ids,
                started_at=time.time(),
            ),
        )

    # ── Polling / waiting ───────────────────────────────────────────

    def poll(self, handle: TargetHandle) -> Literal["pending", "running", "succeeded", "failed"]:
        inner: _LocalInner = handle.inner
        rc = inner.proc.poll()
        if rc is None:
            return "running"
        return "succeeded" if rc == 0 else "failed"

    def wait(self, handle: TargetHandle, timeout: float | None = None) -> dict:
        import json
        inner: _LocalInner = handle.inner
        try:
            rc = inner.proc.wait(timeout=timeout)
        finally:
            # Release locks as soon as the subprocess is done, regardless
            # of exit code.
            if inner.proc.poll() is not None:
                inner.lease.release()
        if rc != 0:
            raise RuntimeError(
                f"local DDP worker exited with rc={rc} "
                f"(cfg={handle.cfg_path})"
            )
        if not handle.result_path.is_file():
            raise RuntimeError(
                f"local DDP worker rc=0 but result missing: {handle.result_path}"
            )
        return json.loads(handle.result_path.read_text())

    def wait_with_pending_timeout(
        self,
        handle: TargetHandle,
        pending_timeout: float,
    ) -> dict:
        # Local jobs are never "pending" in the k8s sense — they start
        # immediately or ``submit`` failed. Delegate to plain ``wait``.
        return self.wait(handle)

    def cancel(self, handle: TargetHandle) -> None:
        inner: _LocalInner = handle.inner
        if inner.proc.poll() is None:
            try:
                inner.proc.terminate()
                try:
                    inner.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    inner.proc.kill()
            finally:
                inner.lease.release()
        else:
            inner.lease.release()


__all__ = ["LocalComputeTarget"]
