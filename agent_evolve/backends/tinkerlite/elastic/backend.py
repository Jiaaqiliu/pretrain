"""``K8sTinkerLiteBackend`` — k8s-first elastic backend (registry key ``k8s_h200``).

Inherits from ``SingleNodeTinkerLiteBackend`` and reuses its pipeline
skeleton (pipeline.yaml → SFT → RL → eval). The *only* substitution is at
the DDP-spawn boundary: instead of a local torchrun subprocess, each stage
is routed through the ``ElasticScheduler``, which picks between k8s and
local targets per availability.

Why inheritance instead of duplication: ``SingleNodeTinkerLiteBackend`` owns
the pipeline orchestration that we don't want to fork. The only seam we need
is how a single DDP stage is executed — and ``single_node.ddp_launcher``
exposes ``override_stage_runner`` exactly for that.

Extras beyond the Protocol (for callers that want parallel fan-out):
``submit_stage_async``, ``wait_any``, ``cancel_stage``. These are the
scheduler's non-blocking API promoted to the backend surface.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ....model.runners.stages.teacher_distill import override_synth_runner
from ....model.types import EvalPlan
from ..single_node import SingleNodeTinkerLiteBackend
from ..single_node.ddp_launcher import override_stage_runner
from .compute_target import CapacityExhausted, ComputeTarget
from .scheduler import ElasticScheduler, FanoutCapacity, StageHandle
from .targets.k8s import K8sComputeTarget
from .targets.local import LocalComputeTarget

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
        # Pod uid/gid — default 1000/1000/1000 matches ec2-user so files
        # created on FSx are writable by the driver process afterwards.
        # Pass None to inherit the image default (root in NGC images).
        k8s_run_as_uid: int | None = 1000,
        k8s_run_as_gid: int | None = 1000,
        k8s_fs_group: int | None = 1000,
        # Local fallback config
        local_enabled: bool = True,
        local_gpu_pool: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
        local_lock_dir: Path = Path("/fsx/.ae_locks"),
        # Scheduler knobs
        queue_timeout_secs: float = 600.0,
        stage_hard_timeout_secs: float | None = None,
        k8s_queue_budget: int = 4,
        # Eval knobs — TP size for the eval pod's vLLM instance. Must
        # match the cluster pod's GPU request. Override with
        # ``AE_K8S_EVAL_GPUS`` if you want a smaller-footprint eval pod.
        eval_world_size: int = 8,
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
                    run_as_uid=k8s_run_as_uid,
                    run_as_gid=k8s_run_as_gid,
                    fs_group=k8s_fs_group,
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
        self.eval_world_size = int(eval_world_size)

    # ── Protocol methods ────────────────────────────────────────────

    def run_trial(self, workspace, node, budget, benchmark):
        """Drop-in compatible with SingleNodeTinkerLiteBackend.run_trial.
        Differences: DDP stages dispatch via the elastic scheduler, and
        teacher-distill (``vllm_local`` provider) dispatches to k8s too."""
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

        def _synth_runner(cfg_path: Path) -> None:
            """Dispatch teacher-distill (vllm_local) to a k8s pod."""
            import json as _json
            cfg = _json.loads(Path(cfg_path).read_text())
            # Inject ``out_result_path`` so the k8s target knows what
            # sentinel to wait on; point it at the stats file the
            # teacher module already writes on success.
            stats_path = Path(cfg["out_path"]).with_suffix(".stats.json")
            cfg["out_result_path"] = str(stats_path)
            Path(cfg_path).write_text(_json.dumps(cfg, indent=2))

            # TP size for the pod's vLLM instance. Teacher YAML carries a
            # ``tensor_parallel_size`` — reuse it; fall back to 8.
            world_size = int(cfg.get("tensor_parallel_size", 8))
            self.scheduler.run_stage(
                cfg_path=Path(cfg_path),
                world_size=world_size,
                log_dir=log_dir,
                stage_label=f"synth-{Path(cfg['out_path']).stem}"[:40]
                            .replace("_", "-"),
                mode="synth",
            )

        with override_stage_runner(_runner), override_synth_runner(_synth_runner):
            return super().run_trial(workspace, node, budget, benchmark)

    # create_training_client / create_sampling_client are inherited
    # unchanged from SingleNodeTinkerLiteBackend.

    # ── Eval (cloudified) ───────────────────────────────────────────

    def run_eval_plan(self, plan: EvalPlan) -> Path:
        """Dispatch the eval through the elastic scheduler.

        Writes a cfg JSON to the plan's output dir on FSx, then submits a
        pod that runs ``agent_evolve.model.runners.eval_worker`` against
        that cfg. Returns the plan's output_dir so the caller's parse
        path keeps working unchanged.

        Why a separate file from the DDP cfg: the eval pod's worker
        takes a different payload shape (``plan`` + ``workspace_root`` +
        ``benchmark_name`` + ``out_result_path``), and reusing the same
        key names as the DDP cfg would be a footgun.
        """
        if self.mock:
            # Preserve the local smoke path for tests — no pod needed.
            return super().run_eval_plan(plan)

        ws_root = Path(
            getattr(self._current_workspace, "root", None)
            or Path(plan.output_dir).resolve().parents[3]
        ).resolve()

        # cfg + sentinel both under the plan's output dir so a failed
        # pod doesn't pollute unrelated paths.
        output_dir = Path(plan.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = output_dir / ".eval_config.json"
        result_path = output_dir / ".eval_result.json"

        benchmark_name = (
            plan.benchmark_name
            or getattr(self._current_benchmark, "name", "nemo_reasoner")
        )

        cfg = {
            "plan": {
                "benchmark_name": plan.benchmark_name,
                "split": plan.split,
                "checkpoint": {
                    "name": plan.checkpoint.name,
                    "path": plan.checkpoint.path,
                    "kind": plan.checkpoint.kind,
                    "metadata": plan.checkpoint.metadata,
                },
                "config_path": plan.config_path,
                "output_dir": plan.output_dir,
                "generation_config": plan.generation_config,
                "metadata": plan.metadata,
            },
            "workspace_root": str(ws_root),
            "benchmark_name": benchmark_name,
            "split": plan.split,
            "out_result_path": str(result_path),
        }
        cfg_path.write_text(json.dumps(cfg, indent=2))

        # Eval stage world_size = TP size for the pod's vLLM instance.
        # Override per-run via AE_K8S_EVAL_GPUS.
        world_size = int(os.environ.get(
            "AE_K8S_EVAL_GPUS", str(self.eval_world_size),
        ))
        log_dir = Path(ws_root) / "logs" / "k8s_stages"
        log_dir.mkdir(parents=True, exist_ok=True)

        stage_label = f"eval-{plan.checkpoint.name}".replace("_", "-")[:40]
        logger.info(
            "[elastic] dispatching eval via k8s (ckpt=%s split=%s world=%d)",
            plan.checkpoint.name, plan.split, world_size,
        )
        self.scheduler.run_stage(
            cfg_path=cfg_path,
            world_size=world_size,
            log_dir=log_dir,
            stage_label=stage_label,
            mode="eval",
        )

        # The scheduler read ``out_result_path`` from the cfg and confirmed
        # the pod exited 0; the pod wrote metrics.json + predictions.jsonl
        # under output_dir directly. Caller parses from there.
        return output_dir

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
