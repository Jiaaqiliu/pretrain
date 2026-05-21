"""SLURM compute backend implementation."""

from __future__ import annotations

import re
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional

from autopilot.backends.base import (
    JobConfig,
    JobHandle,
    JobMetrics,
    JobStatus,
)
from autopilot.utils.logging import get_logger

log = get_logger("backends.slurm")

_SLURM_STATUS_MAP = {
    "PENDING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.COMPLETED,
    "FAILED": JobStatus.FAILED,
    "CANCELLED": JobStatus.CANCELLED,
    "TIMEOUT": JobStatus.FAILED,
    "PREEMPTED": JobStatus.PREEMPTED,
    "NODE_FAIL": JobStatus.FAILED,
}


class SlurmBackend:
    """SLURM compute backend using sbatch/squeue/scancel commands."""

    def __init__(
        self,
        partition: Optional[str] = None,
        account: Optional[str] = None,
        qos: Optional[str] = None,
        default_time_limit: str = "72:00:00",
    ):
        self._partition = partition
        self._account = account
        self._qos = qos
        self._default_time_limit = default_time_limit

    @property
    def name(self) -> str:
        return "slurm"

    def submit_job(self, config: JobConfig) -> JobHandle:
        script = self._build_sbatch_script(config)
        log.info(f"Submitting job '{config.name}' to SLURM")

        result = subprocess.run(
            ["sbatch"],
            input=script,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")

        job_id = self._parse_job_id(result.stdout)
        log.info(f"Job submitted: {job_id}")

        return JobHandle(
            job_id=job_id,
            backend=self.name,
            name=config.name,
            status=JobStatus.PENDING,
            submitted_at=time.time(),
            metadata={"command": config.command},
        )

    def cancel_job(self, handle: JobHandle) -> None:
        log.info(f"Cancelling SLURM job {handle.job_id}")
        subprocess.run(
            ["scancel", handle.job_id],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def get_status(self, handle: JobHandle) -> JobStatus:
        result = subprocess.run(
            ["squeue", "-j", handle.job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            # Job not in queue, check sacct
            result = subprocess.run(
                ["sacct", "-j", handle.job_id, "-n", "-o", "State", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                state = result.stdout.strip().split("\n")[0].split("|")[0]
                return _SLURM_STATUS_MAP.get(state, JobStatus.COMPLETED)
            return JobStatus.COMPLETED

        state = result.stdout.strip()
        return _SLURM_STATUS_MAP.get(state, handle.status)

    def get_logs(self, handle: JobHandle, tail: int = 100) -> str:
        log_file = f"slurm-{handle.job_id}.out"
        result = subprocess.run(
            ["tail", f"-{tail}", log_file],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else ""

    def get_metrics(self, handle: JobHandle) -> Optional[JobMetrics]:
        logs = self.get_logs(handle, tail=50)
        if not logs:
            return None
        return self._parse_metrics_from_logs(logs)

    def list_jobs(
        self, status: Optional[JobStatus] = None, tags: Optional[Dict[str, str]] = None
    ) -> List[JobHandle]:
        cmd = ["squeue", "-u", "$USER", "-h", "-o", "%i|%j|%T"]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, shell=True
        )
        if result.returncode != 0:
            return []

        handles = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                job_status = _SLURM_STATUS_MAP.get(parts[2], JobStatus.PENDING)
                if status and job_status != status:
                    continue
                handles.append(
                    JobHandle(
                        job_id=parts[0],
                        backend=self.name,
                        name=parts[1],
                        status=job_status,
                    )
                )
        return handles

    def get_available_resources(self) -> Dict[str, Any]:
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%P|%a|%D|%t|%G"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}

        resources = {"partitions": []}
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("|")
                if len(parts) >= 5:
                    resources["partitions"].append(
                        {
                            "name": parts[0],
                            "available": parts[1],
                            "nodes": parts[2],
                            "state": parts[3],
                            "gpus": parts[4],
                        }
                    )
        return resources

    def stream_logs(self, handle: JobHandle) -> Iterator[str]:
        log_file = f"slurm-{handle.job_id}.out"
        process = subprocess.Popen(
            ["tail", "-f", log_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            for line in iter(process.stdout.readline, ""):
                yield line.rstrip("\n")
        finally:
            process.terminate()

    def _build_sbatch_script(self, config: JobConfig) -> str:
        lines = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name={config.name}")
        lines.append(f"#SBATCH --nodes={config.num_nodes}")
        lines.append(f"#SBATCH --gpus-per-node={config.num_gpus_per_node}")
        lines.append(f"#SBATCH --ntasks-per-node={config.num_gpus_per_node}")

        if self._partition:
            lines.append(f"#SBATCH --partition={self._partition}")
        if self._account:
            lines.append(f"#SBATCH --account={self._account}")
        if self._qos:
            lines.append(f"#SBATCH --qos={self._qos}")

        time_limit = (
            f"{int(config.timeout_hours)}:00:00"
            if config.timeout_hours
            else self._default_time_limit
        )
        lines.append(f"#SBATCH --time={time_limit}")
        lines.append("#SBATCH --output=slurm-%j.out")
        lines.append("#SBATCH --error=slurm-%j.err")

        lines.append("")
        for key, value in config.env_vars.items():
            lines.append(f"export {key}={value}")

        lines.append("")
        if config.num_nodes > 1:
            torchrun = (
                f"torchrun --nproc-per-node={config.num_gpus_per_node} "
                f"--nnodes={config.num_nodes} "
                f"--rdzv-backend=c10d "
                f"--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT"
            )
            lines.append(f"{torchrun} {' '.join(config.command)}")
        else:
            lines.append(" ".join(config.command))

        return "\n".join(lines)

    def _parse_job_id(self, output: str) -> str:
        match = re.search(r"(\d+)", output)
        return match.group(1) if match else "unknown"

    def _parse_metrics_from_logs(self, logs: str) -> Optional[JobMetrics]:
        metrics = JobMetrics()
        for line in reversed(logs.strip().split("\n")):
            step_match = re.search(r"step[:\s=]+(\d+)", line, re.IGNORECASE)
            loss_match = re.search(r"loss[:\s=]+([\d.]+)", line, re.IGNORECASE)
            if step_match:
                metrics.step = int(step_match.group(1))
            if loss_match:
                metrics.loss = float(loss_match.group(1))
            if metrics.step > 0:
                break
        return metrics if metrics.step > 0 else None
