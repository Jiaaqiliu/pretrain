"""``K8sTinkerLiteBackend`` — k8s-first elastic backend (registry key ``k8s_h200``).

Inherits from ``SingleNodeTinkerLiteBackend`` and reuses its pipeline
skeleton (pipeline.yaml → SFT → RL → eval). The *only* substitution is at
the DDP-spawn boundary: instead of a local torchrun subprocess, each stage
is routed through the ``ElasticScheduler``, which picks between k8s and
local targets per availability.

Why inheritance instead of duplication: ``SingleNodeTinkerLiteBackend`` is
~350 lines of pipeline orchestration that we don't want to fork. The only
seam we need is how a single DDP stage is executed — and ``ddp_launcher``
now exposes ``override_stage_runner`` exactly for that.

Extras beyond the Protocol (for callers that want parallel fan-out):
``submit_stage_async``, ``wait_any``, ``cancel_stage``. These are the
scheduler's non-blocking API promoted to the backend surface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..ddp_launcher import override_stage_runner
from ..single_node import SingleNodeTinkerLiteBackend
from .compute_target import CapacityExhausted, ComputeTarget
from .k8s_target import K8sComputeTarget
from .local_target import LocalComputeTarget
from .scheduler import ElasticScheduler, FanoutCapacity, StageHandle

logger = logging.getLogger(__name__)


class K8sTinkerLiteBackend(SingleNodeTinkerLiteBackend):
    """k8s-first elastic backend. Runs training stages on a shared k8s
    cluster when it can, falls back to the local machine when the cluster
    is saturated or the queue wait is too long.

    When ``local_enabled=False`` (future: no local box), submissions stay
    on k8s and block on the queue indefinitely — no fallback path.
    """

    name = "k8s_h200"

    def __init__(
        self,
        *,
        # K8s config
        namespace: str = "a-evolve",
        image: str = "a-evolve/trainer:latest",
        pvc_name: str = "fsx-zzsamshi",
        pvc_mount_path: str = "/fsx",
        node_selector: dict[str, str] | None = None,
        gpu_resource_key: str = "nvidia.com/gpu",
        kubeconfig: str | None = None,
        ae_root_in_pod: str = "/fsx/zzsamshi/a-evolve",
        k8s_poll_interval_secs: float = 10.0,
        # Local fallback config
        local_enabled: bool = True,
        local_gpu_pool: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
        local_lock_dir: Path = Path("/fsx/.ae_locks"),
        # Scheduler knobs
        queue_timeout_secs: float = 600.0,
        stage_hard_timeout_secs: float | None = None,
        k8s_queue_budget: int = 4,
        # Inherited from SingleNodeTinkerLiteBackend; default False for a
        # real k8s backend — mock doesn't exercise the scheduler.
        mock: bool = False,
    ):
        super().__init__(mock=mock)

        targets: list[ComputeTarget] = []
        try:
            targets.append(
                K8sComputeTarget(
                    namespace=namespace,
                    image=image,
                    pvc_name=pvc_name,
                    pvc_mount_path=pvc_mount_path,
                    node_selector=node_selector,
                    gpu_resource_key=gpu_resource_key,
                    kubeconfig=kubeconfig,
                    ae_root_in_pod=ae_root_in_pod,
                    poll_interval_secs=k8s_poll_interval_secs,
                )
            )
        except RuntimeError as exc:
            # Kubernetes package missing or no kubeconfig. If local is
            # enabled, we can still run; otherwise re-raise.
            if not local_enabled:
                raise
            logger.warning(
                "[k8s_h200] K8s target unavailable (%s); running local-only. "
                "Install `kubernetes` and configure kubeconfig to enable cluster submission.",
                exc,
            )

        if local_enabled:
            targets.append(
                LocalComputeTarget(
                    gpu_pool=local_gpu_pool,
                    lock_dir=local_lock_dir,
                )
            )

        if not targets:
            raise RuntimeError(
                "K8sTinkerLiteBackend has no available compute target "
                "(k8s unavailable and local_enabled=False)"
            )

        self.scheduler = ElasticScheduler(
            targets=targets,
            queue_timeout_secs=queue_timeout_secs,
            stage_hard_timeout_secs=stage_hard_timeout_secs,
            k8s_queue_budget=k8s_queue_budget,
        )

    # ── Protocol methods ────────────────────────────────────────────

    def run_trial(self, workspace, node, budget, benchmark):
        """Drop-in compatible with SingleNodeTinkerLiteBackend.run_trial.
        Only difference: DDP stages dispatch via the elastic scheduler."""
        log_dir = Path(workspace.root) / "logs" / "k8s_stages"
        log_dir.mkdir(parents=True, exist_ok=True)

        def _runner(cfg_path: Path, world_size: int, log_prefix: str) -> None:
            # The scheduler writes result JSON at the path inside cfg; the
            # caller (run_sft_ddp / run_gspo_ddp) reads that file directly
            # after we return. We don't need to return it here.
            self.scheduler.run_stage(
                cfg_path=cfg_path,
                world_size=world_size,
                log_dir=log_dir,
                stage_label=log_prefix,
            )

        with override_stage_runner(_runner):
            return super().run_trial(workspace, node, budget, benchmark)

    # create_training_client / create_sampling_client / run_eval_plan
    # are inherited unchanged from SingleNodeTinkerLiteBackend. eval keeps
    # running on the caller's machine — cloudifying it is a follow-up.

    # ── Non-Protocol extras: parallel fan-out ───────────────────────

    def submit_stage_async(
        self,
        cfg_path: Path,
        world_size: int,
        log_dir: Path,
        *,
        stage_label: str = "stage",
    ) -> StageHandle:
        """Non-blocking: pick a target and submit. Returns a StageHandle
        usable with ``wait_any``."""
        return self.scheduler.submit_async(
            cfg_path, world_size, log_dir, stage_label=stage_label,
        )

    def wait_any(self, handles: list[StageHandle]) -> tuple[StageHandle, dict]:
        """Block until any one of ``handles`` finishes. Returns (handle, result)."""
        return self.scheduler.wait_any(handles)

    def cancel_stage(self, handle: StageHandle) -> None:
        self.scheduler.cancel(handle)

    def probe_fanout_capacity(self, world_size: int) -> FanoutCapacity:
        """Ask the scheduler how many ``world_size``-sized trials it can
        currently absorb. Callers driving parallel sweeps should clamp
        their submission count to ``probe.recommended`` to avoid swamping
        a shared cluster with Pending pods."""
        return self.scheduler.probe_capacity(world_size)


__all__ = [
    "K8sTinkerLiteBackend",
    "CapacityExhausted",
    "StageHandle",
    "FanoutCapacity",
]
