"""Drive a single SFT/GSPO smoke Job through K8sTinkerLiteBackend.

Usage:
    python drive_smoke.py <kind> <cfg_path> [world_size]

This is the host-side tool — it instantiates K8sTinkerLiteBackend, calls
submit_stage_async with the prebuilt cfg_path, then blocks on wait_any.
The actual training runs inside an EKS Job spawned by the backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/fsx/zzsamshi/a-evolve")
sys.path.insert(0, str(REPO))

from agent_evolve.backends.tinkerlite.elastic import K8sTinkerLiteBackend  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    kind = sys.argv[1]
    cfg_path = Path(sys.argv[2])
    world_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 1

    log_dir = cfg_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    backend = K8sTinkerLiteBackend(
        namespace="default",
        image="801953956576.dkr.ecr.ap-southeast-3.amazonaws.com/zzsamshi/a-evolve:5e870af-dirty-20260429042736",
        pvc_name="fsx-pvc",
        pvc_mount_path="/fsx",
        ae_root_in_pod="/fsx/zzsamshi/a-evolve",
        node_selector={"nvidia.com/gpu.product": "NVIDIA-H200"},
        local_enabled=False,    # force k8s-only so we exercise that path
        queue_timeout_secs=1200,
        k8s_poll_interval_secs=8.0,
    )

    cap = backend.probe_fanout_capacity(world_size=world_size)
    print(f"[smoke] fan-out capacity (ws={world_size}): {cap}")

    print(f"[smoke] submitting {kind} Job with cfg={cfg_path}")
    handle = backend.submit_stage_async(
        cfg_path=cfg_path,
        world_size=world_size,
        log_dir=log_dir,
        stage_label=f"smoke-{kind}",
    )
    print(f"[smoke] handle: target={handle.target.name} "
          f"inner={handle.target_handle.inner.job_name}/{handle.target_handle.inner.namespace}")

    sh, result = backend.wait_any([handle])
    print(f"[smoke] done. result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
