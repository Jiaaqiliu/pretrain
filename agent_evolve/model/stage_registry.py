"""Pipeline-stage registry.

Replaces the hard-coded ``if stype == "sft": ... elif stype == "rl": ...``
ladder that used to live in ``SingleNodeTinkerLiteBackend._run_pipeline``.
A new stage type is now a one-file change:

    # runners/stages/my_stage.py
    from agent_evolve.model.stage_registry import register_stage, StageResult

    @register_stage("my_stage")
    def _drive(ctx):
        out_path, stats = do_work(ctx.workspace, ctx.stage, ...)
        return StageResult(checkpoint=None, metrics={"out_path": str(out_path), **stats})

No edits to the backend's dispatcher. See ``INTEGRATION.md`` §2.

The stage adapter receives a ``StageContext`` that bundles everything the
dispatcher used to pass inline (workspace, benchmark, budget_seconds, smoke,
last_ckpt, optimizer, and the training/sampling client factories). It
returns a ``StageResult`` — an optional new ``CheckpointRef`` (stages that
don't produce one, e.g. data generation, return ``None``) plus a metrics
dict that the dispatcher appends to ``aggregated["stage_metrics"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import CheckpointRef


@dataclass
class StageContext:
    """Everything a stage worker needs. Populated by the backend dispatcher."""

    workspace: Any
    stage: dict
    benchmark: Any
    budget_seconds: float | None
    smoke: bool
    last_ckpt: CheckpointRef | None

    # Workspace-scoped config already loaded by the dispatcher (optimizer.yaml,
    # etc.). Stages may read directly from ``workspace`` too — the preloaded
    # version is there for convenience + byte-identical consistency.
    optimizer: dict | None = None

    # Training-client factories. ``training_client_fn`` returns the
    # dispatcher's shared client (built lazily, reused across SFT/RL stages).
    # ``close_training_client_fn`` tears it down — RL calls this before
    # rollout so the vLLM engine gets a whole GPU.
    training_client_fn: Callable[[], Any] | None = None
    close_training_client_fn: Callable[[], None] | None = None

    # Sampling-client factory. Takes a ``CheckpointRef`` (usually the
    # starting adapter for RL rollouts).
    sampling_client_fn: Callable[[CheckpointRef], Any] | None = None

    # Scratch for stages that need to mutate the aggregated report inline
    # (e.g. RL's pre-built metrics dict). Most stages ignore this and return
    # their metrics via ``StageResult``.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """What a stage adapter returns to the dispatcher."""

    # Set if the stage produced a new checkpoint that downstream SFT/RL
    # should consume. Data-gen stages leave this ``None``.
    checkpoint: CheckpointRef | None = None

    # Appended to ``aggregated["stage_metrics"]`` by the dispatcher. The
    # dispatcher will stamp ``stage`` + ``type`` keys if missing.
    metrics: dict[str, Any] = field(default_factory=dict)


StageAdapter = Callable[[StageContext], StageResult]

STAGE_TYPES: dict[str, StageAdapter] = {}


def register_stage(stype: str) -> Callable[[StageAdapter], StageAdapter]:
    """Decorator: register a stage adapter under ``stype``.

    The decorated function must take exactly one ``StageContext`` argument
    and return ``StageResult``. Re-registering the same ``stype`` raises —
    use this as a loud signal that two plugins collided rather than a
    silent last-write-wins.
    """
    def _decorator(fn: StageAdapter) -> StageAdapter:
        if stype in STAGE_TYPES and STAGE_TYPES[stype] is not fn:
            raise RuntimeError(
                f"Stage type {stype!r} already registered "
                f"(existing={STAGE_TYPES[stype]!r}, new={fn!r})."
            )
        STAGE_TYPES[stype] = fn
        return fn
    return _decorator


def resolve_stage(stype: str) -> StageAdapter:
    fn = STAGE_TYPES.get(stype)
    if fn is None:
        raise KeyError(
            f"Unknown stage type: {stype!r}. Registered: {sorted(STAGE_TYPES)}"
        )
    return fn


__all__ = [
    "STAGE_TYPES",
    "StageAdapter",
    "StageContext",
    "StageResult",
    "register_stage",
    "resolve_stage",
]
