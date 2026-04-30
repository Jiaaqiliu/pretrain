"""Agent Evolve -- evolve any agent through a file system contract."""

__version__ = "0.1.0"

from .benchmarks.base import BenchmarkAdapter
from .config import EvolveConfig
from .harness.api import Evolver
from .harness.contract.manifest import Manifest
from .harness.contract.workspace import AgentWorkspace
from .harness.engine.base import EvolutionEngine
from .harness.engine.history import EvolutionHistory
from .harness.engine.trial import TrialRunner
from .harness.protocol.base_agent import BaseAgent
from .training.api import TrainingEvolver
from .training.workspace import TrainingWorkspace
from .types import (
    CycleRecord,
    EvolutionResult,
    Feedback,
    Observation,
    SkillMeta,
    StepResult,
    Task,
    Trajectory,
)

__all__ = [
    "Evolver",
    "TrainingEvolver",
    "TrainingWorkspace",
    "EvolutionEngine",
    "EvolutionHistory",
    "TrialRunner",
    "BaseAgent",
    "BenchmarkAdapter",
    "AgentWorkspace",
    "Manifest",
    "EvolveConfig",
    "Task",
    "Trajectory",
    "Feedback",
    "Observation",
    "SkillMeta",
    "StepResult",
    "CycleRecord",
    "EvolutionResult",
]
