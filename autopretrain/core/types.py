"""Core data types for HoloPretrain."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class JobState(enum.Enum):
    """Formal state machine states for a training job."""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DIAGNOSING = "diagnosing"
    RECOVERING = "recovering"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    ALERT_HUMAN = "alert_human"

    @property
    def is_terminal(self) -> bool:
        return self in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.ALERT_HUMAN,
        )

    @property
    def is_active(self) -> bool:
        return self in (
            JobState.CREATED,
            JobState.PENDING,
            JobState.RUNNING,
            JobState.DIAGNOSING,
            JobState.RECOVERING,
            JobState.TIMED_OUT,
        )


class FailureType(enum.Enum):
    """Classified failure types from multi-layer diagnosis."""

    NETWORK_DISCONNECT = "network_disconnect"
    OOM = "oom"
    NCCL_TIMEOUT = "nccl_timeout"
    DATA_ERROR = "data_error"
    CODE_BUG = "code_bug"
    PREEMPTION = "preemption"
    SILENT_HANG = "silent_hang"
    LOSS_DIVERGENCE = "loss_divergence"
    NAN_DETECTED = "nan_detected"
    STRAGGLER = "straggler"
    NODE_REPEATED_FAIL = "node_repeated_fail"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    UNKNOWN = "unknown"

    @property
    def is_retryable(self) -> bool:
        return self not in (
            FailureType.DATA_ERROR,
            FailureType.CODE_BUG,
        )


class AnomalyType(enum.Enum):
    """Types of training anomalies detected by the monitor."""

    LOSS_SPIKE = "loss_spike"
    GRADIENT_EXPLOSION = "gradient_explosion"
    NAN_DETECTED = "nan_detected"
    THROUGHPUT_DROP = "throughput_drop"
    LOSS_PLATEAU = "loss_plateau"
    DIVERGENCE = "divergence"
    OOM_WARNING = "oom_warning"
    HEARTBEAT_STALE = "heartbeat_stale"
    HEARTBEAT_DEAD = "heartbeat_dead"


class ActionType(enum.Enum):
    """Actions the orchestrator can take."""

    RESUBMIT = "resubmit"
    RESUBMIT_WITH_FIX = "resubmit_with_fix"
    ROLLBACK = "rollback"
    REDUCE_LR = "reduce_lr"
    REDUCE_BATCH = "reduce_batch"
    ENABLE_GRAD_CKPT = "enable_grad_ckpt"
    EARLY_STOP = "early_stop"
    FORCE_KILL = "force_kill"
    EXCLUDE_NODE = "exclude_node"
    WAIT = "wait"
    ALERT_HUMAN = "alert_human"
    NO_ACTION = "no_action"


@dataclass
class MetricSnapshot:
    """A single point-in-time metric observation."""

    step: int
    timestamp: float
    metrics: dict[str, float] = field(default_factory=dict)
    job_id: str = ""

    @property
    def loss(self) -> float | None:
        return self.metrics.get("train/ce_loss")

    @property
    def grad_norm(self) -> float | None:
        return self.metrics.get("optim/grad_norm")

    @property
    def throughput(self) -> float | None:
        return self.metrics.get("throughput/tokens_per_second")

    @property
    def gpu_memory_pct(self) -> float | None:
        return self.metrics.get("gpu_memory/pct")


@dataclass
class Heartbeat:
    """Heartbeat signal from a running training process."""

    trial_id: str
    timestamp: float
    step: int
    loss: float | None = None
    grad_norm: float | None = None
    throughput_tps: float | None = None
    gpu_memory_pct: float | None = None
    status: str = "training"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > 120  # 2 min

    @property
    def is_dead(self) -> bool:
        return self.age_seconds > 360  # 6 min


@dataclass
class Anomaly:
    """A detected training anomaly."""

    type: AnomalyType
    severity: float  # 0.0 to 1.0
    step: int
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Action:
    """An action decided by the orchestrator."""

    type: ActionType
    target_trial_id: str
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecoveryAction:
    """A recovery action selected for a specific failure."""

    action_type: ActionType
    description: str
    config_changes: dict[str, Any] = field(default_factory=dict)
    wait_seconds: int = 0
    confidence: float = 0.9


@dataclass
class DataMixSpec:
    """Specification for a data mixture."""

    sources: dict[str, float]  # domain -> ratio (sums to 1.0)
    paths: dict[str, str] = field(default_factory=dict)  # domain -> glob path
    total_tokens: int | None = None
    max_repetition_ratio: float = 4.0

    def validate(self) -> bool:
        total = sum(self.sources.values())
        return abs(total - 1.0) < 1e-6 and all(v >= 0 for v in self.sources.values())


@dataclass
class TrialConfig:
    """Full configuration for a single training trial."""

    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    data_mix: DataMixSpec | None = None

    # Model
    model_factory: str = "olmo2_1B"
    sequence_length: int = 4096
    vocab_size: int = 100352

    # Training
    max_steps: int = 5000
    global_batch_size: int = 128
    rank_microbatch_size: int = 4
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Resilience
    ephemeral_save_interval: int = 500
    save_interval: int = 5000
    use_skip_step_optimizer: bool = True

    # Infrastructure
    num_gpus: int = 8
    num_nodes: int = 1
    enable_gradient_checkpointing: bool = False

    # Checkpoint resume
    resume_from: str | None = None

    # Runtime tracking
    max_retries: int = 5
    current_retry: int = 0
    deadline_seconds: int = 14400  # 4 hours

    # Paths
    save_folder: str = ""
    code_dir: str = "/fsx/dev/jiaqi/A-EVOLVE-V2"

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    """Result of a completed training trial."""

    trial_id: str
    config: TrialConfig
    final_loss: float | None = None
    final_step: int = 0
    total_tokens: int = 0
    gpu_hours: float = 0.0
    metrics_history: list[MetricSnapshot] = field(default_factory=list)
    eval_results: dict[str, float] = field(default_factory=dict)
    status: str = "completed"  # completed, failed, early_stopped


@dataclass
class Budget:
    """Resource budget for an experiment campaign."""

    max_gpu_hours: float = 100.0
    max_trials: int = 50
    max_retries_total: int = 20
    gpu_hours_used: float = 0.0
    trials_completed: int = 0
    retries_used: int = 0

    @property
    def has_budget(self) -> bool:
        return (
            self.gpu_hours_used < self.max_gpu_hours
            and self.trials_completed < self.max_trials
            and self.retries_used < self.max_retries_total
        )

    @property
    def utilization_pct(self) -> float:
        return (self.gpu_hours_used / self.max_gpu_hours) * 100 if self.max_gpu_hours > 0 else 0

    def consume_gpu_hours(self, hours: float) -> None:
        self.gpu_hours_used += hours

    def consume_retry(self) -> None:
        self.retries_used += 1


@dataclass
class JobEvent:
    """An event in the job lifecycle (for audit trail)."""

    timestamp: float = field(default_factory=time.time)
    trial_id: str = ""
    event_type: str = ""
    from_state: str = ""
    to_state: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""
    evidence: list[str] = field(default_factory=list)
