"""Lightweight mock TrainingClient / SamplingClient for tests and smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....model.types import CheckpointRef
from ..base import (
    AdamParams,
    Datum,
    ForwardBackwardResult,
    Logprobs,
    OptimStepResult,
    Prompt,
    Sample,
    SampleResponse,
    SamplingClient,
    SamplingParams,
    TokenSequence,
    TrainingClient,
)


class MockTrainingClient(TrainingClient):
    def __init__(self, workspace_root: Path):
        self._root = Path(workspace_root)
        self._step = 0

    def forward_backward(
        self,
        batch: list[Datum],
        loss_fn: str,
        loss_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardResult:
        # Loss decreases geometrically so smoke tests can assert progress.
        loss = 1.0 / (self._step + 1)
        return ForwardBackwardResult(loss=loss, extras={"batch_size": len(batch), "loss_fn": loss_fn})  # type: ignore[dict-item]

    def optim_step(self, params: AdamParams) -> OptimStepResult:
        self._step += 1
        return OptimStepResult(step=self._step, learning_rate=params.learning_rate)

    def save_state(self, name: str) -> CheckpointRef:
        path = self._root / "checkpoints" / "full" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "state.json").write_text("{}")
        return CheckpointRef(name=name, path=str(path), kind="full_state")

    def save_weights_for_sampler(self, name: str) -> CheckpointRef:
        path = self._root / "checkpoints" / "sampler" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.json").write_text("{}")
        return CheckpointRef(name=name, path=str(path), kind="sampler_weights")


class MockSamplingClient(SamplingClient):
    def set_prompt_strings(self, prompts: list[Prompt], texts: list[str]) -> None:
        # Mock sampling uses token ids directly, but the real vLLM client
        # requires prompt strings. Keep the protocol surface identical.
        if len(prompts) != len(texts):
            raise ValueError("prompts and texts must have the same length")

    def sample(
        self, prompts: list[Prompt], params: SamplingParams
    ) -> list[SampleResponse]:
        responses: list[SampleResponse] = []
        for p in prompts:
            response = SampleResponse(
                samples=[
                    Sample(tokens=list(p.tokens) + [0] * min(params.max_tokens, 4), text="ok")
                    for _ in range(max(1, params.n))
                ]
            )
            responses.append(response)
        return responses

    def compute_logprobs(self, sequences: list[TokenSequence]) -> list[Logprobs]:
        return [Logprobs(values=[0.0] * len(seq.tokens)) for seq in sequences]
