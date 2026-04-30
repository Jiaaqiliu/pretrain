"""``K8sComputeTarget`` — run a DDP stage as a ``batch/v1`` Job in k8s.

Lazy-imports the ``kubernetes`` client so the rest of this package (and
by extension ``single_node``) doesn't grow a new runtime dependency. If
the dependency is missing we raise with an actionable hint.

Responsibilities:

- ``capacity_probe`` reads node + pod state to decide whether to submit now,
  queue for scheduling, or skip entirely (zero H200 nodes in the cluster).
- ``submit`` writes a Job manifest and creates it. Pod logs get streamed
  into ``log_dir`` by a lightweight background thread so the caller can
  tail them without shelling out to ``kubectl``.
- ``wait_with_pending_timeout`` respects the "pending is ok, running is ok
  indefinitely" policy — a job that never leaves Pending past the timeout
  raises ``PendingTimeout`` so the scheduler can fall back.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from ..compute_target import (
    CapacityReport,
    PendingTimeout,
    TargetHandle,
)
from ..k8s.job_manifest import build_job_manifest

logger = logging.getLogger(__name__)


def _require_kubernetes():
    try:
        from kubernetes import client, config, watch  # noqa: F401
    except ImportError as exc:  # pragma: no cover — exercised in environments w/o k8s dep
        raise RuntimeError(
            "K8sComputeTarget requires the `kubernetes` package. "
            "Install with: pip install 'kubernetes>=31.0'"
        ) from exc
    return client, config, watch


class _K8sInner:
    def __init__(self, job_name: str, namespace: str):
        self.job_name = job_name
        self.namespace = namespace
        self.log_thread: threading.Thread | None = None
        self.log_stop: threading.Event = threading.Event()


class K8sComputeTarget:
    name = "k8s"
    priority = 0  # preferred

    def __init__(
        self,
        *,
        namespace: str = "a-evolve",
        image: str = "a-evolve/trainer:latest",
        pvc_name: str = "fsx-zzsamshi",
        pvc_mount_path: str = "/fsx",
        node_selector: dict[str, str] | None = None,
        gpu_resource_key: str = "nvidia.com/gpu",
        kubeconfig: str | None = None,
        ae_root_in_pod: str = "/fsx/zzsamshi/a-evolve",
        poll_interval_secs: float = 10.0,
    ):
        self.namespace = namespace
        self.image = image
        self.pvc_name = pvc_name
        self.pvc_mount_path = pvc_mount_path
        self.node_selector = dict(node_selector) if node_selector else None
        self.gpu_resource_key = gpu_resource_key
        self.ae_root_in_pod = ae_root_in_pod
        self.poll_interval_secs = float(poll_interval_secs)

        client, config, _ = _require_kubernetes()
        if kubeconfig is not None:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
        self._client = client
        self._v1 = client.CoreV1Api()
        self._batch = client.BatchV1Api()

    # ── Capacity ────────────────────────────────────────────────────

    def capacity_probe(self, required_gpus: int) -> CapacityReport:
        selector = ",".join(f"{k}={v}" for k, v in (self.node_selector or {}).items())
        try:
            nodes = self._v1.list_node(label_selector=selector) if selector else self._v1.list_node()
        except Exception as exc:  # pragma: no cover
            return CapacityReport(False, False, None, f"list_node failed: {exc!r}")

        total_nodes = 0
        total_ready = 0
        free = 0
        for node in nodes.items:
            total_nodes += 1
            if not self._node_ready(node):
                continue
            total_ready += 1
            allocatable = int(node.status.allocatable.get(self.gpu_resource_key, 0))
            used = self._gpus_used_on_node(node.metadata.name)
            free += max(0, allocatable - used)

        return CapacityReport(
            can_run_now=free >= required_gpus,
            can_queue=total_nodes > 0,  # even unready nodes imply a queue exists
            available_gpus=free,
            reason=(
                f"k8s nodes={total_nodes} ready={total_ready} "
                f"free_gpus={free}/{required_gpus} "
                f"selector={selector or 'none'}"
            ),
        )

    @staticmethod
    def _node_ready(node) -> bool:
        for cond in (node.status.conditions or []):
            if cond.type == "Ready":
                return cond.status == "True"
        return False

    def _gpus_used_on_node(self, node_name: str) -> int:
        # Sum GPU requests across non-terminal pods on this node.
        try:
            pods = self._v1.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={node_name}"
            )
        except Exception:  # pragma: no cover
            return 0
        used = 0
        for pod in pods.items:
            if pod.status.phase not in ("Pending", "Running"):
                continue
            for c in (pod.spec.containers or []):
                req = (c.resources.requests or {}) if c.resources else {}
                lim = (c.resources.limits or {}) if c.resources else {}
                raw = req.get(self.gpu_resource_key) or lim.get(self.gpu_resource_key) or 0
                try:
                    used += int(raw)
                except (TypeError, ValueError):
                    pass
        return used

    # ── Submission ──────────────────────────────────────────────────

    def submit(
        self,
        cfg_path: Path,
        world_size: int,
        log_dir: Path,
        *,
        stage_label: str = "stage",
    ) -> TargetHandle:
        import json
        cfg_path = Path(cfg_path)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        cfg = json.loads(cfg_path.read_text())
        result_path = Path(cfg["out_result_path"])

        # Job names need to be DNS-1123 compliant and unique.
        job_name = f"aev-{stage_label}-{uuid.uuid4().hex[:8]}".lower().replace("_", "-")

        manifest = build_job_manifest(
            job_name=job_name,
            namespace=self.namespace,
            image=self.image,
            cfg_path=str(cfg_path),
            world_size=world_size,
            pvc_name=self.pvc_name,
            pvc_mount_path=self.pvc_mount_path,
            ae_root_in_pod=self.ae_root_in_pod,
            node_selector=self.node_selector,
            gpu_resource_key=self.gpu_resource_key,
        )
        self._batch.create_namespaced_job(namespace=self.namespace, body=manifest)
        logger.info("submitted k8s Job %s/%s (ws=%d cfg=%s)",
                    self.namespace, job_name, world_size, cfg_path)

        inner = _K8sInner(job_name=job_name, namespace=self.namespace)
        handle = TargetHandle(
            target_name=self.name,
            cfg_path=cfg_path,
            result_path=result_path,
            inner=inner,
        )
        # Start log tail in background; best-effort, survives pod restarts.
        self._start_log_tail(handle, log_dir / f"{stage_label}.k8s.log")
        return handle

    # ── Polling / waiting ───────────────────────────────────────────

    def poll(self, handle: TargetHandle) -> Literal["pending", "running", "succeeded", "failed"]:
        inner: _K8sInner = handle.inner
        try:
            job = self._batch.read_namespaced_job_status(
                name=inner.job_name, namespace=inner.namespace,
            )
        except Exception:  # pragma: no cover
            return "pending"
        status = job.status or None
        if status is None:
            return "pending"
        if status.succeeded:
            return "succeeded"
        if status.failed:
            return "failed"
        if (status.active or 0) > 0:
            # Pod exists but may still be Pending at the pod level.
            pod_phase = self._lookup_pod_phase(inner.job_name, inner.namespace)
            if pod_phase == "Running":
                return "running"
            return "pending"
        return "pending"

    def _lookup_pod_phase(self, job_name: str, namespace: str) -> str | None:
        try:
            pods = self._v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={job_name}",
            )
        except Exception:  # pragma: no cover
            return None
        if not pods.items:
            return None
        # There should be at most one pod for backoffLimit=0.
        return pods.items[0].status.phase

    def wait(self, handle: TargetHandle, timeout: float | None = None) -> dict:
        import json
        start = time.time()
        while True:
            phase = self.poll(handle)
            if phase == "succeeded":
                break
            if phase == "failed":
                self._stop_log_tail(handle)
                raise RuntimeError(
                    f"k8s Job {handle.inner.job_name} failed (cfg={handle.cfg_path})"
                )
            if timeout is not None and (time.time() - start) >= timeout:
                raise TimeoutError(f"k8s Job {handle.inner.job_name} exceeded timeout {timeout}s")
            time.sleep(self.poll_interval_secs)

        self._stop_log_tail(handle)
        if not handle.result_path.is_file():
            raise RuntimeError(
                f"k8s Job {handle.inner.job_name} succeeded but result missing: {handle.result_path}"
            )
        return json.loads(handle.result_path.read_text())

    def wait_with_pending_timeout(
        self,
        handle: TargetHandle,
        pending_timeout: float,
    ) -> dict:
        """Allow a Job to sit in Pending up to ``pending_timeout`` seconds;
        once it transitions to Running, poll without an upper bound.
        """
        import json
        started_running = False
        pending_start = time.time()
        while True:
            phase = self.poll(handle)
            if phase == "succeeded":
                break
            if phase == "failed":
                self._stop_log_tail(handle)
                raise RuntimeError(
                    f"k8s Job {handle.inner.job_name} failed"
                )
            if phase == "running":
                started_running = True
            elif phase == "pending" and not started_running:
                if (time.time() - pending_start) >= pending_timeout:
                    raise PendingTimeout(
                        f"k8s Job {handle.inner.job_name} still Pending after "
                        f"{pending_timeout}s"
                    )
            time.sleep(self.poll_interval_secs)

        self._stop_log_tail(handle)
        if not handle.result_path.is_file():
            raise RuntimeError(
                f"k8s Job {handle.inner.job_name} succeeded but result missing"
            )
        return json.loads(handle.result_path.read_text())

    def cancel(self, handle: TargetHandle) -> None:
        from kubernetes.client import V1DeleteOptions
        inner: _K8sInner = handle.inner
        self._stop_log_tail(handle)
        try:
            self._batch.delete_namespaced_job(
                name=inner.job_name,
                namespace=inner.namespace,
                body=V1DeleteOptions(propagation_policy="Background"),
            )
            logger.info("canceled k8s Job %s/%s", inner.namespace, inner.job_name)
        except Exception as exc:  # pragma: no cover
            logger.warning("cancel failed for %s: %r", inner.job_name, exc)

    # ── Log tailing ─────────────────────────────────────────────────

    def _start_log_tail(self, handle: TargetHandle, log_path: Path) -> None:
        inner: _K8sInner = handle.inner

        def _tail():
            from kubernetes import watch  # noqa
            fp = open(log_path, "ab", buffering=0)
            try:
                while not inner.log_stop.is_set():
                    pods = self._v1.list_namespaced_pod(
                        namespace=inner.namespace,
                        label_selector=f"job-name={inner.job_name}",
                    )
                    if not pods.items:
                        time.sleep(2.0)
                        continue
                    pod_name = pods.items[0].metadata.name
                    try:
                        stream = self._v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=inner.namespace,
                            follow=True,
                            _preload_content=False,
                        )
                        for chunk in stream.stream(amt=4096):
                            if inner.log_stop.is_set():
                                break
                            if isinstance(chunk, str):
                                chunk = chunk.encode()
                            fp.write(chunk)
                    except Exception:
                        time.sleep(2.0)
                        continue
                    # If we got here the log stream ended — pod likely done.
                    if self.poll(handle) in ("succeeded", "failed"):
                        break
            finally:
                fp.close()

        t = threading.Thread(target=_tail, name=f"k8s-logs-{inner.job_name}", daemon=True)
        inner.log_thread = t
        t.start()

    def _stop_log_tail(self, handle: TargetHandle) -> None:
        inner: _K8sInner = handle.inner
        inner.log_stop.set()
        if inner.log_thread is not None:
            inner.log_thread.join(timeout=5.0)


__all__ = ["K8sComputeTarget"]
