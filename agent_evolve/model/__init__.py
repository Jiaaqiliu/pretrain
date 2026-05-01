"""Training evolution subsystem for A-Evolve."""

from .api import TrainingEvolver
from .runner_protocol import TrainingJobRunner
from .types import (
    CheckpointRef,
    ErrorBuckets,
    EvalMetrics,
    EvalPlan,
    MCGSCycleReport,
    MetricSpec,
    PatchOperation,
    TrainingEvolutionResult,
    TrainingEvolveConfig,
    TrainingRegistryError,
    TrainingSearchNode,
    TrainingSearchNodeSummary,
    TrainingTrialResult,
    TrainingWorkspaceNotFound,
    TrainingWorkspaceValidationError,
    TrialBudget,
    ValidityReport,
    WorkspaceFingerprint,
    WorkspaceMutation,
    WorkspacePatch,
)
from .workspace import TrainingWorkspace

__all__ = [
    "TrainingEvolver",
    "TrainingJobRunner",
    "TrainingWorkspace",
    "TrainingEvolveConfig",
    "TrainingEvolutionResult",
    "TrainingTrialResult",
    "TrainingSearchNode",
    "TrainingSearchNodeSummary",
    "MCGSCycleReport",
    "CheckpointRef",
    "EvalMetrics",
    "EvalPlan",
    "ErrorBuckets",
    "MetricSpec",
    "PatchOperation",
    "TrialBudget",
    "ValidityReport",
    "WorkspaceFingerprint",
    "WorkspaceMutation",
    "WorkspacePatch",
    "TrainingRegistryError",
    "TrainingWorkspaceNotFound",
    "TrainingWorkspaceValidationError",
]
