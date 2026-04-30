"""Model-adapter registry — plug a new fine-tuning surface (DoRA, IA³, full
fine-tune, QLoRA) without editing ``clients/hf.py`` or ``ddp_worker.py``.

Ships only ``LoRAAdapter`` initially (preserves today's behavior). Adding a
new adapter is one file + one ``@register_adapter("<kind>")`` decorator,
then set ``model/adapter.yaml::type: <kind>`` in your seed workspace.

See ``INTEGRATION.md`` §3 for a DoRA example.
"""

from .base import (
    ModelAdapter,
    ATTACH_MODE_WRAP,
    ATTACH_MODE_INPLACE,
    register_adapter,
    resolve_adapter,
    ADAPTERS,
)
from .lora import LoRAAdapter

__all__ = [
    "ModelAdapter",
    "ATTACH_MODE_WRAP",
    "ATTACH_MODE_INPLACE",
    "LoRAAdapter",
    "register_adapter",
    "resolve_adapter",
    "ADAPTERS",
]
