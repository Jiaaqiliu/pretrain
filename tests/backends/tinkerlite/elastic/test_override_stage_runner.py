"""Verify ``override_stage_runner`` actually intercepts the DDP spawn.

This is the only seam between single_node's pipeline code and the k8s
backend's scheduler. If it breaks, the k8s backend silently runs local
torchrun subprocesses — which is the exact bug we're avoiding.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_evolve.backends.tinkerlite.single_node import ddp_launcher as ddp_mod
from agent_evolve.backends.tinkerlite.single_node.ddp_launcher import (
    override_stage_runner,
    run_sft_ddp,
)


class _WS:
    def __init__(self, root: Path):
        self.root = str(root)


def test_override_routes_sft_ddp_through_hook(tmp_path: Path, monkeypatch) -> None:
    # Prevent the default _spawn_torchrun from ever running.
    called_default = []
    monkeypatch.setattr(
        ddp_mod, "_spawn_torchrun",
        lambda *a, **kw: called_default.append(True),
    )

    captured = []

    def fake_runner(cfg_path, world_size, log_prefix):
        captured.append((Path(cfg_path), world_size, log_prefix))
        # Worker's job is to produce the result file.
        cfg = json.loads(Path(cfg_path).read_text())
        result_path = Path(cfg["out_result_path"])
        result_path.write_text(json.dumps({"opt_steps": 42, "wall_seconds": 1.0}))

    ws = _WS(tmp_path)
    base_cfg = {"path": "/fsx/models/X"}
    adapter_cfg = {"rank": 16}
    optimizer_cfg = {"lr": 5e-5}
    batching_cfg = {"per_device_bs": 1, "grad_accum": 8, "max_seq_len": 2560}
    stage = {"name": "sft_warmup"}

    with override_stage_runner(fake_runner):
        ckpt, stats = run_sft_ddp(
            ws, stage,
            base_cfg=base_cfg, adapter_cfg=adapter_cfg,
            optimizer_cfg=optimizer_cfg, batching_cfg=batching_cfg,
            world_size=8,
        )

    # Default was NOT called; our override was.
    assert not called_default
    assert len(captured) == 1
    cfg_path, world_size, log_prefix = captured[0]
    assert world_size == 8
    assert log_prefix == "sft-ddp"
    # The cfg file was actually written for the override to read.
    assert cfg_path.is_file()

    # Result propagated back.
    assert stats["opt_steps"] == 42


def test_override_is_scoped(tmp_path: Path, monkeypatch) -> None:
    """After exiting the context manager, default spawn is restored."""
    default_calls = []
    monkeypatch.setattr(
        ddp_mod, "_spawn_torchrun",
        lambda cfg, ws, log_prefix: (
            default_calls.append(cfg),
            Path(json.loads(Path(cfg).read_text())["out_result_path"]).write_text(
                '{"opt_steps": 0}'
            ),
        ),
    )

    ws = _WS(tmp_path)
    base_cfg = {"path": "/fsx/models/X"}
    adapter_cfg = {"rank": 16}
    optimizer_cfg = {"lr": 5e-5}
    batching_cfg = {"per_device_bs": 1, "grad_accum": 8, "max_seq_len": 2560}
    stage = {"name": "sft_warmup"}

    with override_stage_runner(lambda *a, **kw: None):
        pass  # no-op inside; leave scope cleanly

    # Outside the context, default spawn should run.
    run_sft_ddp(
        ws, stage,
        base_cfg=base_cfg, adapter_cfg=adapter_cfg,
        optimizer_cfg=optimizer_cfg, batching_cfg=batching_cfg,
        world_size=4,
    )
    assert len(default_calls) == 1
