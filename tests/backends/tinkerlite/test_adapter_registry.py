"""PR-C targeted tests: ``ModelAdapter`` Protocol + registry.

The registry reserves the extension point for non-LoRA adapters (DoRA,
QLoRA, full-weight, IA³, custom head). Today only ``LoRAAdapter`` ships
— this test fleet just verifies the seam exists and behaves.
"""

from __future__ import annotations

import pytest

from agent_evolve.backends.tinkerlite.adapters import (
    ADAPTERS,
    ATTACH_MODE_WRAP,
    LoRAAdapter,
    ModelAdapter,
    register_adapter,
    resolve_adapter,
)
from agent_evolve.training.types import CheckpointRef


def test_lora_is_registered_on_import():
    assert "lora" in ADAPTERS
    a = resolve_adapter("lora")
    assert isinstance(a, LoRAAdapter)
    assert a.kind == "lora"
    assert a.attach_mode == ATTACH_MODE_WRAP


def test_lora_adapter_conforms_to_protocol():
    # Runtime Protocol conformance check — no torch/PEFT import needed.
    assert isinstance(LoRAAdapter(), ModelAdapter)


def test_resolve_unknown_adapter_raises():
    with pytest.raises(KeyError, match="Unknown adapter kind"):
        resolve_adapter("not_an_adapter_kind")


def test_checkpoint_ref_kind_literal_widened_to_full_weights():
    """``CheckpointRef.kind`` now accepts ``full_weights`` (for non-LoRA
    SFT) in addition to ``adapter``."""
    c = CheckpointRef(name="x", path="/tmp/x", kind="full_weights")
    assert c.kind == "full_weights"


def test_register_new_adapter_via_decorator():
    class _FakeAdapter:
        kind = "ut_fake_adapter"
        attach_mode = ATTACH_MODE_WRAP

        def attach(self, base, cfg):
            return base

        def save(self, model, tokenizer, outdir):
            return CheckpointRef(name=outdir.name, path=str(outdir), kind="full_weights")

        def vllm_lora_request(self, checkpoint):
            return None  # full-weight sampler

    register_adapter("ut_fake_adapter")(_FakeAdapter)
    try:
        a = resolve_adapter("ut_fake_adapter")
        assert isinstance(a, ModelAdapter)
        assert a.vllm_lora_request(CheckpointRef(name="x", path="/tmp/x")) is None
    finally:
        ADAPTERS.pop("ut_fake_adapter", None)
