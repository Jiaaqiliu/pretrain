"""TinkerLite — local training-backend abstraction shaped like Tinker."""

from .base import (
    AdamParams,
    Datum,
    ForwardBackwardResult,
    Logprobs,
    ModelInput,
    OptimStepResult,
    Prompt,
    Sample,
    SampleResponse,
    SamplingClient,
    SamplingParams,
    TinkerLiteBackend,
    TokenSequence,
    TrainingClient,
)


def __getattr__(name: str):
    # Lazy imports: the real clients pull in torch / vllm, which the smoke +
    # unit-test paths don't need. Importing them lazily keeps ``from
    # agent_evolve.backends.tinkerlite import Datum`` cheap.
    if name == "HFTrainingClient":
        from .hf_clients import HFTrainingClient as _HFTrainingClient

        return _HFTrainingClient
    if name == "build_hf_client_from_workspace":
        from .hf_clients import build_hf_client_from_workspace as _build

        return _build
    if name == "VLLMSamplingClient":
        from .vllm_sampling import VLLMSamplingClient as _VLLMSamplingClient

        return _VLLMSamplingClient
    raise AttributeError(f"module 'agent_evolve.backends.tinkerlite' has no attribute {name!r}")


__all__ = [
    "TinkerLiteBackend",
    "TrainingClient",
    "SamplingClient",
    "Datum",
    "ModelInput",
    "SamplingParams",
    "AdamParams",
    "ForwardBackwardResult",
    "OptimStepResult",
    "SampleResponse",
    "Sample",
    "TokenSequence",
    "Prompt",
    "Logprobs",
    "HFTrainingClient",
    "VLLMSamplingClient",
    "build_hf_client_from_workspace",
]
