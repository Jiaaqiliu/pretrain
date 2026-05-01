"""TinkerLite backend protocol + datatypes.

Shape mirrors Tinker's quickstart (https://tinker-docs.thinkingmachines.ai/).
Kept **sync**, unlike Tinker's async API, because TinkerLite is an
in-process execution layer — no RPC boundary to justify futures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ...model.types import (
    CheckpointRef,
    EvalPlan,
    TrainingSearchNode,
    TrainingTrialResult,
    TrialBudget,
)


# ── Training datatypes ───────────────────────────────────────────────────

@dataclass
class ModelInput:
    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        return cls(tokens=list(tokens))


@dataclass
class Datum:
    model_input: ModelInput
    loss_fn_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdamParams:
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class ForwardBackwardResult:
    loss: float
    extras: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimStepResult:
    step: int
    learning_rate: float
    extras: dict[str, float] = field(default_factory=dict)


# ── Sampling datatypes ───────────────────────────────────────────────────

@dataclass
class Prompt:
    tokens: list[int]


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    n: int = 1


@dataclass
class Sample:
    tokens: list[int]
    text: str | None = None
    logprob: float | None = None


@dataclass
class SampleResponse:
    samples: list[Sample]


@dataclass
class TokenSequence:
    tokens: list[int]


@dataclass
class Logprobs:
    values: list[float]


# ── Client protocols ─────────────────────────────────────────────────────

@runtime_checkable
class TrainingClient(Protocol):
    def forward_backward(
        self,
        batch: list[Datum],
        loss_fn: str,
        loss_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardResult: ...

    def optim_step(self, params: AdamParams) -> OptimStepResult: ...

    def save_state(self, name: str) -> CheckpointRef: ...

    def save_weights_for_sampler(self, name: str) -> CheckpointRef: ...


@runtime_checkable
class SamplingClient(Protocol):
    def set_prompt_strings(self, prompts: list[Prompt], texts: list[str]) -> None: ...

    def sample(
        self, prompts: list[Prompt], params: SamplingParams
    ) -> list[SampleResponse]: ...

    def compute_logprobs(self, sequences: list[TokenSequence]) -> list[Logprobs]: ...


# ── Backend protocol ─────────────────────────────────────────────────────

# ``TinkerLiteBackend`` is the LLM-shaped extension of the generic
# ``TrainingJobRunner`` Protocol: it adds client-factory methods so stages
# can construct HF/PEFT training clients and vLLM sampling clients. Non-LLM
# jobs (sklearn, JAX, tabular) should implement just ``TrainingJobRunner``
# from ``training.runner_protocol`` and ignore this.
from ...model.runner_protocol import TrainingJobRunner  # noqa: E402


@runtime_checkable
class TinkerLiteBackend(TrainingJobRunner, Protocol):
    """LLM-specific extension of ``TrainingJobRunner``.

    Same ``run_trial`` contract plus HF/vLLM client factories and the
    ``run_eval_plan`` entrypoint. Kept as a separate Protocol so stage
    workers under ``training/runners/stages/`` can type-hint against the
    richer surface without pinning non-LLM runners to LLM semantics.
    """

    def create_training_client(
        self,
        workspace: Any,
        checkpoint: CheckpointRef | None = None,
    ) -> TrainingClient: ...

    def create_sampling_client(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
    ) -> SamplingClient: ...

    def run_eval_plan(self, plan: EvalPlan) -> Path: ...
