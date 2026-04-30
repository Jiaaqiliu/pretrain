"""``FullDeepspeedAdapter`` — Protocol conformance + behavioral contract.

The full-param adapter is the first ``ATTACH_MODE_INPLACE`` plugin: it
proves the ``ModelAdapter`` extension point handles non-LoRA fine-tuning
surfaces without changes to the runner / backend / eval layers.

These tests pin the contract that ``runners/stages/sft.py`` and
``runners/stages/eval.py`` rely on:

* ``attach_mode == ATTACH_MODE_INPLACE`` — flips SFT onto the HF Trainer
  path.
* ``vllm_lora_request(ckpt) is None`` — flips eval onto the full-state
  vLLM path.
* ``save`` returns ``CheckpointRef(kind="full_state")`` — what eval keys
  off.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_evolve.backends.tinkerlite.adapters import (
    ADAPTERS,
    ATTACH_MODE_INPLACE,
    FullDeepspeedAdapter,
    ModelAdapter,
    resolve_adapter,
)
from agent_evolve.training.types import CheckpointRef


def test_full_deepspeed_is_registered_on_import():
    assert "full_deepspeed_customized" in ADAPTERS
    a = resolve_adapter("full_deepspeed_customized")
    assert isinstance(a, FullDeepspeedAdapter)
    assert a.kind == "full_deepspeed_customized"
    assert a.attach_mode == ATTACH_MODE_INPLACE


def test_full_deepspeed_conforms_to_protocol():
    assert isinstance(FullDeepspeedAdapter(), ModelAdapter)


def test_attach_is_inplace_no_wrap():
    """``attach`` must not return a wrapper — it returns the same object,
    only with ``use_cache`` flipped and gradient checkpointing on."""
    adapter = FullDeepspeedAdapter()
    base = SimpleNamespace(
        config=SimpleNamespace(use_cache=True),
        gradient_checkpointing_enable=MagicMock(),
    )
    out = adapter.attach(base, cfg={"type": "full_deepspeed_customized"})
    assert out is base, "ATTACH_MODE_INPLACE must return the base model unchanged"
    assert base.config.use_cache is False
    base.gradient_checkpointing_enable.assert_called_once()


def test_save_writes_full_state_checkpoint(tmp_path: Path):
    """``save`` must call ``save_pretrained`` on both model and tokenizer
    and return ``CheckpointRef(kind="full_state")``."""
    adapter = FullDeepspeedAdapter()
    outdir = tmp_path / "ckpt"
    outdir.mkdir()
    (outdir / "config.json").write_text("{}")  # simulate save_pretrained side-effect

    model = MagicMock()
    tokenizer = MagicMock()
    ckpt = adapter.save(model, tokenizer, outdir)

    model.save_pretrained.assert_called_once_with(str(outdir))
    tokenizer.save_pretrained.assert_called_once_with(str(outdir))
    assert ckpt.kind == "full_state"
    assert ckpt.path == str(outdir)
    assert ckpt.metadata.get("adapter_type") == "full_deepspeed_customized"


def test_save_raises_when_config_json_missing(tmp_path: Path):
    """If ``save_pretrained`` doesn't emit ``config.json`` the saved dir
    is unusable for vLLM — fail loudly instead of returning a bad ref."""
    adapter = FullDeepspeedAdapter()
    outdir = tmp_path / "ckpt"
    outdir.mkdir()
    # No config.json written here.

    model = MagicMock()
    tokenizer = MagicMock()
    with pytest.raises(RuntimeError, match="config.json"):
        adapter.save(model, tokenizer, outdir)


def test_vllm_lora_request_returns_none():
    """No ``LoRARequest`` for full-state checkpoints — vLLM loads
    ``checkpoint.path`` directly as the model. ``runners/stages/eval.py``
    keys off this to skip the LoRA adapter wiring."""
    adapter = FullDeepspeedAdapter()
    ckpt = CheckpointRef(name="x", path="/tmp/x", kind="full_state")
    assert adapter.vllm_lora_request(ckpt) is None
