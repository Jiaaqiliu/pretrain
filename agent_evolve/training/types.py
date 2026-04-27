"""Shared dataclasses and exceptions for the TrainingEvolver subsystem.

This module is the single source of truth for every type the training
package exchanges across its boundaries (API facade, workspace, backend,
benchmark, MCGS). No other module in `agent_evolve.training` should
re-define these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ── Exceptions ───────────────────────────────────────────────────────────

class TrainingWorkspaceNotFound(FileNotFoundError):
    """Raised when the workspace path passed to TrainingEvolver does not exist."""


class TrainingWorkspaceValidationError(ValueError):
    """Raised when the workspace layout violates the training contract."""

    def __init__(self, root: str, errors: list[str]):
        self.root = root
        self.errors = errors
        super().__init__(f"Invalid training workspace at {root}: {errors}")


class TrainingRegistryError(ValueError):
    """Raised when a string key cannot be resolved via a training registry."""


# ── Config ───────────────────────────────────────────────────────────────

@dataclass
class TrainingEvolveConfig:
    max_cycles: int = 20
    smoke: bool = False
    trial_budget_seconds: float = 600.0
    trial_budget_steps: int | None = None
    trial_budget_tokens: int | None = None
    # Free-form namespace for algorithm/backend/benchmark specific knobs.
    extra: dict[str, Any] = field(default_factory=dict)


# ── Workspace / mutation ─────────────────────────────────────────────────

@dataclass
class PatchOperation:
    op: Literal["replace", "add", "remove"]
    path: str
    key_path: list[str] | None = None
    value: Any = None


@dataclass
class WorkspacePatch:
    operations: list[PatchOperation] = field(default_factory=list)


@dataclass
class WorkspaceMutation:
    mutation_id: str
    parent_node_id: str
    description: str
    patch: WorkspacePatch
    mutation_type: Literal[
        "data_mix",
        "training_recipe",
        "loss",
        "reward",
        "rollout",
        "pipeline",
        "fusion",
        "debug",
    ] = "data_mix"


@dataclass
class WorkspaceFingerprint:
    workspace_name: str
    manifest_hash: str
    evolvable_hash: str
    protected_hash: str
    git_sha: str | None
    created_at: str


# ── Backend / training ───────────────────────────────────────────────────

@dataclass
class TrialBudget:
    seconds: float | None = None
    steps: int | None = None
    tokens: int | None = None


@dataclass
class CheckpointRef:
    """Pointer to a saved set of weights / adapter on disk."""

    name: str
    path: str
    kind: Literal["full_state", "sampler_weights", "adapter"] = "adapter"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    primary_metric_name: str
    primary_metric_value: float
    maximize: bool = True
    secondary: dict[str, float] = field(default_factory=dict)


@dataclass
class ErrorBuckets:
    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[dict]] = field(default_factory=dict)


@dataclass
class ValidityReport:
    is_valid: bool
    hard_fail_reason: str | None = None
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSpec:
    name: str
    maximize: bool = True
    higher_is_better: bool = True


@dataclass
class EvalPlan:
    benchmark_name: str
    split: str
    checkpoint: CheckpointRef
    config_path: str
    output_dir: str
    generation_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


TrialStatus = Literal[
    "success",
    "train_failed",
    "eval_failed",
    "invalid_adapter",
    "over_budget",
]


@dataclass
class TrainingTrialResult:
    node_id: str
    workspace_path: str
    status: TrialStatus = "success"
    checkpoint: CheckpointRef | None = None

    eval_metrics: EvalMetrics | None = None
    error_buckets: ErrorBuckets | None = None
    validity: ValidityReport | None = None

    train_metrics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


# ── MCGS nodes / reports ─────────────────────────────────────────────────

@dataclass
class TrainingSearchNode:
    node_id: str
    parent_id: str | None
    branch_id: int

    mutation_plan: str = ""
    workspace_patch: WorkspacePatch = field(default_factory=WorkspacePatch)
    pipeline_summary: str = ""

    trial_status: str | None = None
    metric: float | None = None
    reward: float | None = None

    validity: ValidityReport | None = None
    error_buckets: dict[str, int] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)

    checkpoint: CheckpointRef | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    visits: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0

    is_valid: bool | None = None
    is_terminal: bool = False

    local_best_node_id: str | None = None
    continue_improve: bool = False


@dataclass
class TrainingSearchNodeSummary:
    node_id: str
    parent_id: str | None
    branch_id: int
    metric: float | None
    reward: float | None
    is_valid: bool | None
    visits: int
    mean_reward: float


@dataclass
class MCGSCycleReport:
    cycle: int
    selected_parent_id: str | None
    trial_node_ids: list[str]
    incumbent_node_id: str | None
    incumbent_changed: bool
    best_metric: float | None
    graph_path: str
    report_path: str


# ── Top-level result ─────────────────────────────────────────────────────

@dataclass
class TrainingEvolutionResult:
    workspace: str
    cycles_completed: int
    incumbent_node_id: str | None
    best_metric: float | None
    best_checkpoint: CheckpointRef | None
    graph_path: str
    report_path: str
    topk: list[TrainingSearchNodeSummary] = field(default_factory=list)
