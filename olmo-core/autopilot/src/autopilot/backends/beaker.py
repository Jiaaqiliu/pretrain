"""Beaker compute backend implementation.

Wraps OLMo-core's BeakerLaunchConfig to provide a uniform job management interface.
"""

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

log = get_logger("backends.beaker")

_BEAKER_STATUS_MAP = {
    "created": JobStatus.PENDING,
    "submitted": JobStatus.QUEUED,
    "scheduled": JobStatus.QUEUED,
    "running": JobStatus.RUNNING,
    "succeeded": JobStatus.COMPLETED,
    "failed": JobStatus.FAILED,
    "stopped": JobStatus.CANCELLED,
    "cancelled": JobStatus.CANCELLED,
    "preempted": JobStatus.PREEMPTED,
}


class BeakerBackend:
    """Beaker compute backend using the beaker CLI and OLMo-core launch utilities."""

    def __init__(
        self,
        default_image: Optional[str] = None,
        default_cluster: Optional[str] = None,
        workspace: Optional[str] = None,
    ):
        self._default_image = default_image
        self._default_cluster = default_cluster
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "beaker"

    def submit_job(self, config: JobConfig) -> JobHandle:
        cmd = self._build_beaker_command(config)
        log.info(f"Submitting job '{config.name}' to Beaker")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"Beaker submission failed: {result.stderr}")

        job_id = self._parse_experiment_id(result.stdout)
        log.info(f"Job submitted: {job_id}")

        return JobHandle(
            job_id=job_id,
            backend=self.name,
            name=config.name,
            status=JobStatus.PENDING,
            cluster=self._default_cluster,
            submitted_at=time.time(),
            metadata={"command": config.command, "config": config.__dict__},
        )

    def cancel_job(self, handle: JobHandle) -> None:
        log.info(f"Cancelling job {handle.job_id}")
        subprocess.run(
            ["beaker", "experiment", "stop", handle.job_id],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def get_status(self, handle: JobHandle) -> JobStatus:
        result = subprocess.run(
            ["beaker", "experiment", "get", handle.job_id, "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return handle.status

        import json

        try:
            data = json.loads(result.stdout)
            if isinstance(data, list):
                data = data[0]
            beaker_status = data.get("status", {}).get("current", "unknown")
            return _BEAKER_STATUS_MAP.get(beaker_status, handle.status)
        except (json.JSONDecodeError, KeyError, IndexError):
            return handle.status

    def get_logs(self, handle: JobHandle, tail: int = 100) -> str:
        result = subprocess.run(
            ["beaker", "experiment", "logs", handle.job_id, f"--tail={tail}"],
            capture_output=True,
            text=True,
            timeout=30,
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
        cmd = ["beaker", "experiment", "list", "--format=json"]
        if self._workspace:
            cmd.extend(["--workspace", self._workspace])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []

        import json

        try:
            experiments = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        handles = []
        for exp in experiments:
            exp_status = _BEAKER_STATUS_MAP.get(
                exp.get("status", {}).get("current", "unknown"), JobStatus.PENDING
            )
            if status and exp_status != status:
                continue
            handles.append(
                JobHandle(
                    job_id=exp.get("id", ""),
                    backend=self.name,
                    name=exp.get("name", ""),
                    status=exp_status,
                )
            )
        return handles

    def get_available_resources(self) -> Dict[str, Any]:
        result = subprocess.run(
            ["beaker", "cluster", "utilization", self._default_cluster or "", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"available": True}

        import json

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"available": True}

    def stream_logs(self, handle: JobHandle) -> Iterator[str]:
        process = subprocess.Popen(
            ["beaker", "experiment", "logs", handle.job_id, "--follow"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            for line in iter(process.stdout.readline, ""):
                yield line.rstrip("\n")
        finally:
            process.terminate()

    def _build_beaker_command(self, config: JobConfig) -> List[str]:
        cmd = ["beaker", "experiment", "create"]

        if config.name:
            cmd.extend(["--name", config.name])
        if self._workspace:
            cmd.extend(["--workspace", self._workspace])

        image = config.image or self._default_image
        if image:
            cmd.extend(["--image", image])

        if config.num_gpus_per_node > 0:
            cmd.extend(["--gpus", str(config.num_gpus_per_node)])

        if config.priority:
            cmd.extend(["--priority", config.priority])

        if config.preemptible:
            cmd.append("--preemptible")

        if self._default_cluster:
            cmd.extend(["--cluster", self._default_cluster])

        for key, value in config.env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.append("--")
        cmd.extend(config.command)
        return cmd

    def _parse_experiment_id(self, output: str) -> str:
        match = re.search(r"(ex_[a-zA-Z0-9]+|[0-9a-f]{24})", output)
        if match:
            return match.group(1)
        lines = output.strip().split("\n")
        return lines[-1].strip() if lines else "unknown"

    def _parse_metrics_from_logs(self, logs: str) -> Optional[JobMetrics]:
        metrics = JobMetrics()
        lines = logs.strip().split("\n")

        for line in reversed(lines):
            if "step" in line.lower() or "loss" in line.lower():
                step_match = re.search(r"step[:\s=]+(\d+)", line, re.IGNORECASE)
                loss_match = re.search(r"loss[:\s=]+([\d.]+)", line, re.IGNORECASE)
                lr_match = re.search(r"lr[:\s=]+([\d.e\-+]+)", line, re.IGNORECASE)
                grad_match = re.search(r"grad[_\s]?norm[:\s=]+([\d.]+)", line, re.IGNORECASE)

                if step_match:
                    metrics.step = int(step_match.group(1))
                if loss_match:
                    metrics.loss = float(loss_match.group(1))
                if lr_match:
                    metrics.learning_rate = float(lr_match.group(1))
                if grad_match:
                    metrics.grad_norm = float(grad_match.group(1))

                if metrics.step > 0:
                    break

        return metrics if metrics.step > 0 else None
