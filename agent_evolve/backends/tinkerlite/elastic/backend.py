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


def _preempt_local_gpus() -> None:
    """SIGKILL any CURRENT-USER python process currently attached to the
    host's GPUs. Called before every local eval so lingering vLLM
    servers / abandoned training clients don't OOM the fresh run.

    Idempotent + safe:
      - Filters to current uid via ``os.getuid()`` — never touches other
        users' processes.
      - Skips this process itself (``os.getpid()``) and its ancestors
        so the driver doesn't self-kill.
      - Uses ``nvidia-smi --query-compute-apps`` which reports only
        processes with GPU context attached (not every python proc).
      - All steps are best-effort; failure to preempt is logged and the
        eval continues. It will fail loudly on OOM if a leaker slipped
        through, which is better than silently blocking on a lock.
    """
    import subprocess
    import signal

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.info("[preempt] nvidia-smi unavailable (%s); skipping GPU preempt", exc)
        return

    if result.returncode != 0:
        logger.info("[preempt] nvidia-smi rc=%d; skipping", result.returncode)
        return

    my_pid = os.getpid()
    try:
        # Full ancestry: we don't want to kill our parent shell / pytest / driver.
        ancestors = {my_pid}
        pid = my_pid
        while pid > 1:
            try:
                with open(f"/proc/{pid}/stat") as f:
                    pid = int(f.read().split()[3])
                ancestors.add(pid)
            except (FileNotFoundError, ValueError, IndexError):
                break
    except Exception:  # noqa: BLE001
        ancestors = {my_pid}

    my_uid = os.getuid()
    killed = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 1:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in ancestors:
            continue
        # Only this user's processes.
        try:
            if os.stat(f"/proc/{pid}").st_uid != my_uid:
                continue
        except (FileNotFoundError, PermissionError):
            continue
        pname = parts[1] if len(parts) > 1 else "?"
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append((pid, pname))
        except (ProcessLookupError, PermissionError) as exc:
            logger.info("[preempt] SIGTERM pid=%d failed: %s", pid, exc)

    if not killed:
        return

    logger.info("[preempt] SIGTERM'd %d GPU procs: %s", len(killed), killed)
    # Give them 5s to exit cleanly, then SIGKILL survivors.
    import time
    time.sleep(5)
    for pid, pname in killed:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info("[preempt] SIGKILL pid=%d (%s)", pid, pname)
        except ProcessLookupError:
            pass


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
        """Run the eval LOCALLY on the driver host.

        The k8s path dispatched eval to a pod for historical reasons
        (the cluster has 8×H200s and we were spreading load), but every
        GPU-accelerated path on these k8s nodes currently fails: the
        cluster driver is 570.148.08 while torch 2.10 ships triton 3.6
        which emits CUDA-13-era PTX. Driver 570 rejects the kernel image
        for every triton JIT it sees (vLLM, torch.compile, mamba-ssm).
        The driver host runs driver 580.126 and the same kernels load
        fine there, so we route eval back to ``super().run_eval_plan``.

        Before launching, we preempt any process holding the local GPUs
        (lingering vLLM from the previous cycle, abandoned training
        clients) so eval doesn't OOM against leaked state. Training
        still goes through k8s because DDP doesn't trigger triton JIT.
        """
        _preempt_local_gpus()
        return super().run_eval_plan(plan)

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
