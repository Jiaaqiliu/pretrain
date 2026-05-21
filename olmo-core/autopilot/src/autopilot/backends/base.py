"""Abstract compute backend interface.

All job scheduling systems (Beaker, SLURM, Kubernetes) implement this interface,
allowing the agent to submit, monitor, and manage training jobs uniformly.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable


class JobStatus(enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


@dataclass
class JobConfig:
    """Configuration for submitting a training job."""

    name: str
    command: List[str]
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    image: Optional[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    preemptible: bool = False
    max_retries: int = 0
    timeout_hours: Optional[float] = None
    working_dir: Optional[str] = None
    mounts: Dict[str, str] = field(default_factory=dict)  # host_path -> container_path
    resources: Dict[str, Any] = field(default_factory=dict)  # backend-specific resources
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class JobHandle:
    """Reference to a submitted job."""

    job_id: str
    backend: str
    name: str
    status: JobStatus = JobStatus.PENDING
    cluster: Optional[str] = None
    submitted_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobMetrics:
    """Real-time metrics from a running job."""

    step: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    grad_norm: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    elapsed_seconds: float = 0.0
    tokens_seen: int = 0
    custom: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol for compute backend implementations."""

    @property
    def name(self) -> str:
        """Backend identifier (e.g., 'beaker', 'slurm', 'kubernetes')."""
        ...

    def submit_job(self, config: JobConfig) -> JobHandle:
        """Submit a training job and return a handle."""
        ...

    def cancel_job(self, handle: JobHandle) -> None:
        """Cancel a running or queued job."""
        ...

    def get_status(self, handle: JobHandle) -> JobStatus:
        """Get current job status."""
        ...

    def get_logs(self, handle: JobHandle, tail: int = 100) -> str:
        """Get recent log lines from the job."""
        ...

    def get_metrics(self, handle: JobHandle) -> Optional[JobMetrics]:
        """Get current training metrics from the job."""
        ...

    def list_jobs(
        self, status: Optional[JobStatus] = None, tags: Optional[Dict[str, str]] = None
    ) -> List[JobHandle]:
        """List jobs matching filters."""
        ...

    def get_available_resources(self) -> Dict[str, Any]:
        """Query available compute resources (GPUs, nodes, etc.)."""
        ...

    def stream_logs(self, handle: JobHandle) -> Iterator[str]:
        """Stream log lines from a running job."""
        ...
