"""Parallel LR sweep using ``K8sTinkerLiteBackend``'s fan-out API.

Contrast with ``drive_lr_sweep_4cycle.py`` which runs the same 4 LRs
serially through MCGS. Here we bypass MCGS and submit all four SFT
stages concurrently — k8s absorbs whichever ones fit cluster capacity,
the rest queue or fall back to local. Then we evaluate the resulting
adapters serially on the local machine (vLLM can't share GPUs cleanly).

Not intended to be a canonical usage — MCGS-driven flows are still
preferred for search. This is a demonstration of the
``submit_stage_async`` / ``wait_any`` extension API for callers that
want explicit parallelism.

Launch:
    .../python examples/nemo_reasoning_example/drive_k8s_lr_sweep.py
Outputs:
    $AE/runs/k8s-lr-sweep/<lr>/...
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

AE = Path("/fsx/zzsamshi/a-evolve")
sys.path.insert(0, str(AE))

from agent_evolve.backends.tinkerlite.common_cfg import (  # noqa: E402
    build_sft_cfg,
    default_world_size,
)
from agent_evolve.backends.tinkerlite.k8s import K8sTinkerLiteBackend  # noqa: E402


LRS = (1e-4, 5e-5, 3e-5, 1e-5)
SEED_WORKSPACE = AE / "seed_workspaces" / "nemotron_reasoner"
RUN_ROOT = AE / "runs" / "k8s-lr-sweep"


class _WorkspaceShim:
    """Minimal workspace stand-in accepted by ``build_sft_cfg``."""
    def __init__(self, root: Path):
        self.root = str(root)


def _fork_workspace(seed: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(seed, dst, symlinks=True)
    return dst


def _load_yaml(path: Path) -> dict:
    import yaml
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _patch_lr(workspace_root: Path, lr: float) -> None:
    """Mutate ``train/optimizer.yaml::lr`` to ``lr``."""
    import yaml
    opt_path = workspace_root / "train" / "optimizer.yaml"
    cfg = _load_yaml(opt_path)
    cfg["lr"] = float(lr)
    opt_path.write_text(yaml.safe_dump(cfg))


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    backend = K8sTinkerLiteBackend(
        namespace=os.environ.get("AE_K8S_NAMESPACE", "a-evolve"),
        image=os.environ.get("AE_K8S_IMAGE", "a-evolve/trainer:latest"),
        pvc_name=os.environ.get("AE_K8S_PVC", "fsx-zzsamshi"),
        node_selector=(
            {"nvidia.com/gpu.product": "H200"}
            if os.environ.get("AE_K8S_NODE_LABEL", "1") == "1"
            else None
        ),
        local_enabled=os.environ.get("AE_K8S_LOCAL_ENABLED", "1") == "1",
        queue_timeout_secs=float(os.environ.get("AE_K8S_QUEUE_TIMEOUT", "600")),
    )

    world_size = int(os.environ.get("AE_WORLD_SIZE", str(default_world_size())))

    # Ask the scheduler how many trials we can safely fan out right now.
    # Cap at len(LRS) — no point oversubscribing beyond the sweep size.
    cap = backend.probe_fanout_capacity(world_size)
    max_inflight = min(len(LRS), max(1, cap.recommended))
    print(
        f"[driver] fan-out capacity: recommended={cap.recommended} "
        f"breakdown={cap.breakdown}; using max_inflight={max_inflight} of {len(LRS)} LRs",
        flush=True,
    )
    print(f"[driver] capacity reason: {cap.reason}", flush=True)

    # Stage 1: fan out SFT Jobs, respecting max_inflight. When a handle
    # completes we immediately submit the next LR, keeping the in-flight
    # count saturated until all LRs are issued.
    pending_lrs = list(LRS)
    inflight: list[tuple[float, object]] = []   # (lr, StageHandle)
    results: dict[float, dict] = {}

    def _submit_one(lr: float) -> None:
        ws_root = RUN_ROOT / f"lr-{lr:.0e}"
        _fork_workspace(SEED_WORKSPACE, ws_root / "workspace")
        ws = _WorkspaceShim(ws_root / "workspace")
        _patch_lr(Path(ws.root), lr)

        outdir = Path(ws.root) / "checkpoints" / "adapters" / "sft_warmup"
        outdir.mkdir(parents=True, exist_ok=True)
        cfg_path = outdir / ".ddp_config.json"
        result_path = outdir / ".ddp_result.json"

        stage = {"name": "sft_warmup", "epochs": 2, "loss": "cross_entropy"}
        base_cfg = _load_yaml(Path(ws.root) / "model" / "base.yaml")
        adapter_cfg = _load_yaml(Path(ws.root) / "model" / "adapter.yaml")
        optimizer_cfg = _load_yaml(Path(ws.root) / "train" / "optimizer.yaml")
        batching_cfg = _load_yaml(Path(ws.root) / "train" / "batching.yaml")

        sft_cfg = build_sft_cfg(
            ws, stage,
            base_cfg=base_cfg, adapter_cfg=adapter_cfg,
            optimizer_cfg=optimizer_cfg, batching_cfg=batching_cfg,
            outdir=outdir, result_path=result_path,
        )
        cfg_path.write_text(json.dumps(sft_cfg, indent=2))

        log_dir = ws_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handle = backend.submit_stage_async(
            cfg_path=cfg_path, world_size=world_size, log_dir=log_dir,
            stage_label=f"sft-lr-{lr:.0e}",
        )
        print(
            f"[driver] submitted lr={lr:.0e} target={handle.target.name} "
            f"(inflight={len(inflight)+1}/{max_inflight})",
            flush=True,
        )
        inflight.append((lr, handle))

    # Initial fill.
    while pending_lrs and len(inflight) < max_inflight:
        _submit_one(pending_lrs.pop(0))

    # Drain + refill as each one finishes.
    while inflight:
        handle, result = backend.wait_any([h for _, h in inflight])
        for i, (lr, h) in enumerate(inflight):
            if h is handle:
                results[lr] = result
                inflight.pop(i)
                print(
                    f"[driver] done lr={lr:.0e} target={handle.target.name} "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
                break
        if pending_lrs:
            _submit_one(pending_lrs.pop(0))

    # Stage 3 (not parallelized): evaluate each adapter serially on local vLLM.
    # Left as an exercise — reuse ``backend.run_eval_plan`` or the existing
    # MCGS-driven driver.

    print("\n=== SWEEP COMPLETE ===")
    for lr in LRS:
        r = results.get(lr, {})
        print(f"  lr={lr:.0e}: opt_steps={r.get('opt_steps', '?')} wall={r.get('wall_seconds', '?'):.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
