"""SFT training worker.

In smoke mode the worker uses :class:`MockTrainingClient` to avoid loading any
real model. PR7+ will add a real HF/LoRA path guarded by ``if not smoke``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ...backends.tinkerlite.base import AdamParams, Datum, ModelInput
from ...backends.tinkerlite.mock_clients import MockTrainingClient
from ..types import CheckpointRef


def run_sft_stage(
    workspace: Any,
    stage: dict,
    datums: Iterable[Datum],
    *,
    optimizer: dict | None = None,
    smoke: bool = True,
    budget_seconds: float | None = None,
) -> tuple[CheckpointRef, dict[str, Any]]:
    """Run one SFT stage.

    Returns the checkpoint plus a metrics dict (avg loss, steps, etc.).
    """
    import time

    if smoke:
        client: Any = MockTrainingClient(Path(workspace.root))
    else:
        # Real path lives behind a TODO until PR7 lands the real backend.
        raise NotImplementedError("Non-smoke SFT path is not available in this PR")

    loss_fn = stage.get("loss", "cross_entropy")
    lr = (optimizer or {}).get("lr", 1e-4)
    total_loss = 0.0
    total_steps = 0
    start = time.time()

    # Materialize the iterable so we can optionally repeat it up to ``steps``.
    batch = list(datums) or [Datum(model_input=ModelInput.from_ints([0]))]
    steps = int(stage.get("steps", 1))
    for step_idx in range(steps):
        if budget_seconds is not None and (time.time() - start) > budget_seconds:
            break
        result = client.forward_backward(batch, loss_fn)
        total_loss += result.loss
        client.optim_step(AdamParams(learning_rate=lr))
        total_steps += 1

    ckpt = client.save_weights_for_sampler(name=stage.get("name", f"stage_{total_steps}"))
    return ckpt, {
        "total_steps": total_steps,
        "avg_loss": total_loss / max(1, total_steps),
        "stage": stage.get("name"),
        "loss_fn": loss_fn,
    }
