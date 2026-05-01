"""``LoRAAdapter`` — PEFT LoRA implementation of ``ModelAdapter``.

Captures the exact behavior baked into ``clients/hf.py:85-134, 191-205`` and
``ddp_worker.py:96-105, 146-153`` so callers that migrate to
``resolve_adapter("lora").attach(...)`` get byte-identical results.

Intentionally thin — all LoRA knobs live in ``model/adapter.yaml``; the
adapter reads them and ignores anything it doesn't know.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....model.types import CheckpointRef
from .base import ATTACH_MODE_WRAP, ModelAdapter, register_adapter


@register_adapter("lora")
class LoRAAdapter:
    """PEFT LoRA. The default; what everything does today."""

    kind = "lora"
    attach_mode = ATTACH_MODE_WRAP

    def attach(self, base_model: Any, cfg: dict) -> Any:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=int(cfg.get("rank", 16)),
            lora_alpha=int(cfg.get("alpha", 32)),
            lora_dropout=float(cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(
                cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"],
                )
            ),
        )
        return get_peft_model(base_model, lora_cfg)

    def save(self, model: Any, tokenizer: Any, outdir: Path) -> CheckpointRef:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(outdir))
        tokenizer.save_pretrained(str(outdir))
        if not (outdir / "adapter_config.json").is_file():
            raise RuntimeError(
                f"LoRA save_pretrained did not emit adapter_config.json under {outdir}"
            )
        return CheckpointRef(
            name=outdir.name,
            path=str(outdir),
            kind="adapter",
            metadata={},
        )

    def vllm_lora_request(self, checkpoint: CheckpointRef) -> Any:
        from vllm.lora.request import LoRARequest

        return LoRARequest(checkpoint.name or "candidate", 1, checkpoint.path)


__all__ = ["LoRAAdapter"]
