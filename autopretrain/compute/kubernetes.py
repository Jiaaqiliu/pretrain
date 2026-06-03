"""Kubernetes compute backend.

Manages K8s Jobs for training trials via kubectl.
Supports:
- Job submission (apply manifest)
- Job deletion
- Status polling (job conditions + pod phase)
- Log retrieval
- Node identification (for node exclusion)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from autopretrain.core.types import TrialConfig

logger = logging.getLogger(__name__)


@dataclass
class K8sConfig:
    """K8s backend configuration."""

    namespace: str = "default"
    context: str = "arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm"
    kubectl_timeout: int = 30


class KubernetesBackend:
    """K8s compute backend using kubectl subprocess calls."""

    def __init__(self, config: K8sConfig | None = None) -> None:
        self.config = config or K8sConfig()
        self._ctx_args = ["--context", self.config.context]
        self._ns_args = ["-n", self.config.namespace]

    async def _run(self, cmd: list[str], input_data: str | None = None) -> tuple[int, str, str]:
        """Run a kubectl command asynchronously."""
        full_cmd = ["kubectl"] + cmd + self._ns_args + self._ctx_args
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_data.encode() if input_data else None),
            timeout=self.config.kubectl_timeout,
        )
        return proc.returncode or 0, stdout.decode(), stderr.decode()

    async def submit_job(self, trial: TrialConfig, manifest: str) -> bool:
        """Submit a K8s Job from a YAML manifest string."""
        code, stdout, stderr = await self._run(["apply", "-f", "-"], input_data=manifest)
        if code != 0:
            logger.error("Failed to submit job: %s", stderr)
            return False
        logger.info("Job submitted: %s", stdout.strip())
        return True

    async def delete_job(self, job_name: str) -> bool:
        """Delete a K8s Job."""
        code, _, stderr = await self._run(["delete", "job", job_name, "--ignore-not-found"])
        return code == 0

    async def get_job_status(self, job_name: str) -> str:
        """Get job status: Running, Complete, Failed, Pending, NotFound."""
        # Check job conditions
        code, stdout, _ = await self._run([
            "get", "job", job_name,
            "-o", "jsonpath={.status.conditions[-1:].type}",
        ])

        status = stdout.strip()
        if status in ("Complete", "Failed"):
            return status

        # Check pod phases
        code, stdout, _ = await self._run([
            "get", "pods", "-l", f"job-name={job_name}",
            "-o", "jsonpath={.items[*].status.phase}",
        ])
        phases = stdout.strip().split()

        if "Running" in phases:
            return "Running"
        elif "Pending" in phases:
            return "Pending"
        elif not phases:
            return "NotFound"

        return "Running"

    async def get_pod_logs(self, job_name: str, tail: int = 200) -> str:
        """Get recent logs from a job's pod."""
        code, stdout, stderr = await self._run([
            "logs", f"job/{job_name}", "--tail", str(tail),
        ])
        return stdout if code == 0 else stderr

    async def get_pod_events(self, job_name: str) -> str:
        """Get K8s events for a job."""
        code, stdout, _ = await self._run([
            "get", "events",
            "--field-selector", f"involvedObject.name={job_name}",
        ])
        return stdout

    async def get_node_name(self, job_name: str) -> str | None:
        """Get the node a job's pod is running on."""
        code, stdout, _ = await self._run([
            "get", "pods", "-l", f"job-name={job_name}",
            "-o", "jsonpath={.items[0].spec.nodeName}",
        ])
        node = stdout.strip()
        return node if node else None
