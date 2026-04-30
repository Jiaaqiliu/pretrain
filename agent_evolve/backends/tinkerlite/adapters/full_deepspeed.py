"""``FullDeepspeedAdapter`` — full-parameter SFT via HF Trainer + DeepSpeed.

Counterpart to :class:`LoRAAdapter`. Where LoRA wraps the base model with a
trainable adapter and runs through the step-driven
``forward_backward + optim_step`` loop, full-param training:

* leaves the base model unmodified (no PEFT wrapping),
* drives training through HF :class:`~transformers.Trainer` (so
  ``deepspeed`` ZeRO-3 can shard parameters / gradients / optimizer state),
* persists a complete ``model.safetensors`` checkpoint,
* signals to the eval path (via ``CheckpointRef.kind == "full_state"``)
  that vLLM should load the saved checkpoint *as the model itself* —
  no ``LoRARequest``.

This is the first :class:`ModelAdapter` plugin with
``attach_mode == ATTACH_MODE_INPLACE``. The SFT stage runner uses that
flag to switch from the LoRA step-driven path to the Trainer-driven
path; nothing else in the system needs to know about full-param.

Trigger contract: set ``model/adapter.yaml::type: full_deepspeed_customized``
in your seed workspace. See ``seed_workspaces/posttrain_bench/`` for a
working example and ``INTEGRATION.md §3`` for the general extension
recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....training.types import CheckpointRef
from .base import ATTACH_MODE_INPLACE, register_adapter


@register_adapter("full_deepspeed_customized")
class FullDeepspeedAdapter:
    """Full-parameter SFT, no PEFT wrapping. Runs via HF Trainer +
    optional DeepSpeed ZeRO-3."""

    kind = "full_deepspeed_customized"
    attach_mode = ATTACH_MODE_INPLACE

    def attach(self, base_model: Any, cfg: dict) -> Any:
        """Prepare the base model for full-param training.

        We deliberately do **not** wrap with PEFT; ``base_model`` is the
        thing that will train. The only state mutations are the standard
        memory-saving knobs HF Trainer expects to be flipped before
        ``trainer.train()``: disable KV cache and turn on gradient
        checkpointing.
        """
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        return base_model

    def save(self, model: Any, tokenizer: Any, outdir: Path) -> CheckpointRef:
        """Persist the full model + tokenizer.

        HF :meth:`Trainer.save_model` already does this when called with
        an explicit path; we re-call ``save_pretrained`` to make this
        method usable from non-Trainer callers (mocks, custom loops,
        manual snapshots) without breaking the Trainer path — repeated
        ``save_pretrained`` is idempotent.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(outdir))
        tokenizer.save_pretrained(str(outdir))
        # Validity guard: the saved dir must contain a config.json so the
        # eval path can load it back. Don't enforce a specific weights
        # filename — sharded full models emit ``model-00001-of-N.safetensors``
        # and a ``model.safetensors.index.json`` instead of a single file.
        if not (outdir / "config.json").is_file():
            raise RuntimeError(
                f"FullDeepspeedAdapter.save did not emit config.json under {outdir}"
            )
        return CheckpointRef(
            name=outdir.name,
            path=str(outdir),
            kind="full_state",
            metadata={"adapter_type": self.kind},
        )

    def vllm_lora_request(self, checkpoint: CheckpointRef) -> Any | None:
        """No LoRA — vLLM should load ``checkpoint.path`` as the model."""
        return None


__all__ = ["FullDeepspeedAdapter"]
