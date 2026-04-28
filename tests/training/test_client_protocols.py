"""Protocol conformance tests for the TinkerLite TrainingClient / SamplingClient.

Real-client smoke (HFTrainingClient / VLLMSamplingClient) requires torch+vllm
and is skipped when those aren't importable. What we always verify:

  * MockTrainingClient and MockSamplingClient satisfy the runtime_checkable
    Protocols declared in base.py. If someone tweaks the protocol, this test
    fails loudly before any downstream runner does.
  * Real-client modules import cleanly when torch/vllm are available.
"""

from __future__ import annotations

import importlib.util

import pytest

from agent_evolve.backends.tinkerlite.base import (
    SamplingClient,
    TrainingClient,
)
from agent_evolve.backends.tinkerlite.mock_clients import (
    MockSamplingClient,
    MockTrainingClient,
)


def test_mock_training_client_satisfies_protocol(tmp_path) -> None:
    client = MockTrainingClient(tmp_path)
    assert isinstance(client, TrainingClient)


def test_mock_sampling_client_satisfies_protocol() -> None:
    client = MockSamplingClient()
    assert isinstance(client, SamplingClient)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("peft") is None
    or importlib.util.find_spec("transformers") is None,
    reason="torch+peft+transformers not installed",
)
def test_hf_training_client_module_importable() -> None:
    from agent_evolve.backends.tinkerlite import hf_clients  # noqa: F401

    # The class exists and exposes the protocol methods.
    cls = hf_clients.HFTrainingClient
    for name in ("forward_backward", "optim_step", "save_weights_for_sampler"):
        assert callable(getattr(cls, name))


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vllm not installed",
)
def test_vllm_sampling_client_module_importable() -> None:
    from agent_evolve.backends.tinkerlite import vllm_sampling  # noqa: F401

    cls = vllm_sampling.VLLMSamplingClient
    for name in ("sample", "compute_logprobs"):
        assert callable(getattr(cls, name))
