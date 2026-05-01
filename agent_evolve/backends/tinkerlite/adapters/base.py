"""``ModelAdapter`` Protocol — how a trainable surface attaches to a base model.

Separates three concerns that today are baked into ``clients/hf.py`` and
``ddp_worker.py``:

  1. ``attach`` — wrap the base model with the trainable surface (LoRA
     adapter, DoRA, full-weight unfreeze, prefix tuning, ...).
  2. ``save``   — persist the trainable surface to an outdir, returning a
     ``CheckpointRef`` whose ``kind`` field tells the sampler how to load
     it (``"adapter"`` vs ``"full_weights"``).
  3. ``vllm_lora_request`` — build the per-sampler request object. For
     non-LoRA adapters (full-weight, custom head), return ``None`` and the
     sampler loads the saved checkpoint as a full model.

``clients/hf.py`` and ``ddp_worker.py`` today implement only the LoRA case;
``LoRAAdapter`` in this package captures that exact behavior so nothing
changes at runtime. Future kinds drop in as additional modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from ....model.types import CheckpointRef


# ``attach_mode`` signals how the training loop must treat the return value
# of ``attach``. Kept as a string constant rather than an enum so plugin
# modules don't need to import a shared enum just to declare themselves.
ATTACH_MODE_WRAP = "wrap"        # attach returns a new wrapper model (PEFT-style)
ATTACH_MODE_INPLACE = "inplace"  # attach modifies base in place (full-weight unfreeze)


@runtime_checkable
class ModelAdapter(Protocol):
    """Pluggable fine-tuning surface.

    Implementations are small: 3 methods + 2 class attributes. Register
    via ``@register_adapter("<kind>")`` and reference by
    ``model/adapter.yaml::type``.
    """

    kind: str        # value of ``adapter.yaml::type`` this implementation handles
    attach_mode: str # ATTACH_MODE_WRAP or ATTACH_MODE_INPLACE

    def attach(self, base_model: Any, cfg: dict) -> Any:
        """Return the model ready for ``.train()``.

        ``cfg`` is the loaded ``model/adapter.yaml`` dict — implementations
        read their own knobs (``rank``, ``alpha``, ``target_modules``, ...)
        and ignore the rest.
        """

    def save(self, model: Any, tokenizer: Any, outdir: Path) -> CheckpointRef:
        """Persist to ``outdir``. Returns a ``CheckpointRef`` — the caller
        uses ``kind`` to decide how the sampler should load this.
        """

    def vllm_lora_request(self, checkpoint: CheckpointRef) -> Any | None:
        """Build the vLLM ``LoRARequest`` for this checkpoint, or ``None``
        if the sampler should load ``checkpoint.path`` as a full model
        (no LoRA)."""


ADAPTERS: dict[str, ModelAdapter] = {}


def register_adapter(kind: str) -> Callable[[type], type]:
    """Decorator: register a ``ModelAdapter`` class under ``kind``.

    We store an instance (not the class) so ``resolve_adapter`` returns
    something the caller can immediately call methods on. Implementations
    that need init args should register an instance explicitly with
    ``ADAPTERS[kind] = MyAdapter(...)`` instead of using the decorator.
    """
    def _decorator(cls: type) -> type:
        if kind in ADAPTERS and type(ADAPTERS[kind]) is not cls:
            raise RuntimeError(
                f"Model adapter {kind!r} already registered "
                f"(existing={type(ADAPTERS[kind]).__name__}, new={cls.__name__})."
            )
        ADAPTERS[kind] = cls()
        return cls
    return _decorator


def resolve_adapter(kind: str) -> ModelAdapter:
    adapter = ADAPTERS.get(kind)
    if adapter is None:
        raise KeyError(
            f"Unknown adapter kind: {kind!r}. Registered: {sorted(ADAPTERS)}"
        )
    return adapter


__all__ = [
    "ATTACH_MODE_INPLACE",
    "ATTACH_MODE_WRAP",
    "ADAPTERS",
    "ModelAdapter",
    "register_adapter",
    "resolve_adapter",
]
