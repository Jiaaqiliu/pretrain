"""Builder for ``batch/v1`` Job manifests that run a-evolve workers.

Two modes, selected via ``mode``:

* ``"ddp"`` (default) — wraps the command in ``torch.distributed.run`` and
  runs ``agent_evolve.model.runners.ddp_worker``. Single-pod DDP
  (``nproc_per_node=world_size``); cross-node DDP needs PyTorchJob/MPIJob
  and is out of scope.
* ``"eval"`` — runs ``agent_evolve.model.runners.eval_worker`` directly
  (vLLM spawns its own TP workers; no torchrun). ``world_size`` is the
  GPU count requested for the pod (TP size).

Design choices:

- Code lives on an FSx PVC mounted at ``/fsx`` in the pod; we do NOT bake
  code into the image. This lets iteration be "save file → resubmit Job"
  with no rebuild.
- ``restartPolicy=Never`` + ``backoffLimit=0`` — we want exactly-once
  execution; the scheduler retries explicitly, not k8s.
"""

from __future__ import annotations

from typing import Any, Literal


def build_job_manifest(
    *,
    job_name: str,
    namespace: str,
    image: str,
    cfg_path: str,               # absolute path inside the pod (under /fsx)
    world_size: int,
    pvc_name: str,
    pvc_mount_path: str = "/fsx",
    ae_root_in_pod: str = "/fsx/zzsamshi/a-evolve",
    node_selector: dict[str, str] | None = None,
    gpu_resource_key: str = "nvidia.com/gpu",
    extra_env: dict[str, str] | None = None,
    shm_size_gib: int = 16,
    labels: dict[str, str] | None = None,
    run_as_uid: int | None = 1000,
    run_as_gid: int | None = 1000,
    fs_group: int | None = 1000,
    mode: Literal["ddp", "eval"] = "ddp",
) -> dict[str, Any]:
    """Return a plain-dict Job manifest; caller feeds it to ``kubernetes.client``.

    ``cfg_path`` / adapter dir / result file must all live under
    ``pvc_mount_path`` so that rank 0 inside the pod writes to the same
    FSx bytes that the scheduler reads on the host.

    By default the pod runs as uid/gid 1000 so any files it creates on
    the shared FSx PVC are readable + writable by the ec2-user on the
    host that continues driving the cycle after the Job completes (e.g.
    the post-training local eval step that writes under
    ``evolution/eval/``). Pass ``run_as_uid=None`` / ``run_as_gid=None``
    to disable and inherit the image's default user.
    """
    master_port = 29500 + (hash(cfg_path) % 1000)

    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_DISABLED": "true",
        "NCCL_DEBUG": "WARN",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": ae_root_in_pod,
        # uid 1000 has no /etc/passwd entry in the NGC image, so HOME
        # is unset. Point it at a writable path to keep stray ~/foo
        # cache writes from failing.
        "HOME": "/tmp",
    }
    # vLLM on this cluster hits flashinfer compile errors; the eval
    # and synth (teacher_distill vllm_local) pods both spin up vLLM,
    # so bake the disable flags into their env by default. DDP pods
    # don't import vllm, so skip.
    if mode in ("eval", "synth"):
        env.update({
            "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0",
            "VLLM_USE_FLASHINFER_MOE_FP8": "0",
            "VLLM_USE_FLASHINFER_MOE_FP4": "0",
            "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
            # On EKS nodes with driver 570 + torch 2.10, Inductor's pre-cached
            # static CUDA kernels fail to load with "device kernel image is
            # invalid" during vLLM profile_run. Disable the FX graph + static
            # launcher caches so triton JITs kernels fresh against THIS
            # node's driver/device. Cache dir gets a unique per-pod path
            # (the `$HOSTNAME` is k8s pod name) so we never pick up stale bits.
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TORCHINDUCTOR_CACHE_DIR": f"/tmp/torchinductor-{job_name}",
            "TRITON_CACHE_DIR": f"/tmp/triton-{job_name}",
            "VLLM_CACHE_ROOT": f"/tmp/vllm-cache-{job_name}",
            # Also kill the prebuilt static cuda launcher path that hosts
            # the exact "load_kernel" line crashing today.
            "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER": "0",
        })
    if extra_env:
        env.update(extra_env)

    if mode == "ddp":
        command = [
            "python",
            "-m", "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            "-m", "agent_evolve.model.runners.ddp_worker",
            "--config", cfg_path,
        ]
        container_name = "trainer"
    elif mode == "eval":
        # vLLM spawns its own TP workers; no torchrun required.
        command = [
            "python",
            "-m", "agent_evolve.model.runners.eval_worker",
            "--config", cfg_path,
        ]
        container_name = "evaluator"
    elif mode == "synth":
        # Teacher-distill vllm_local: same module used for local subprocess,
        # just invoked inside a pod instead of on the host.
        command = [
            "python",
            "-m", "agent_evolve.model.runners.stages.teacher_distill",
            "--config", cfg_path,
        ]
        container_name = "teacher"
    else:
        raise ValueError(f"unknown manifest mode: {mode!r}")

    container: dict[str, Any] = {
        "name": container_name,
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "workingDir": ae_root_in_pod,
        "command": command,
        "env": [{"name": k, "value": v} for k, v in env.items()],
        "resources": {
            "limits": {gpu_resource_key: str(world_size)},
            "requests": {gpu_resource_key: str(world_size)},
        },
        "volumeMounts": [
            {"name": "fsx", "mountPath": pvc_mount_path},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ],
    }

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": [
            {
                "name": "fsx",
                "persistentVolumeClaim": {"claimName": pvc_name},
            },
            {
                "name": "dshm",
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": f"{shm_size_gib}Gi",
                },
            },
        ],
    }
    if node_selector:
        pod_spec["nodeSelector"] = dict(node_selector)

    # Ensure files created on FSx are owned by the host's ec2-user
    # (uid/gid 1000 by default) rather than root, so the host-side
    # eval step can read + write under the pod's output paths.
    security_context: dict[str, Any] = {}
    if run_as_uid is not None:
        security_context["runAsUser"] = int(run_as_uid)
    if run_as_gid is not None:
        security_context["runAsGroup"] = int(run_as_gid)
    if fs_group is not None:
        security_context["fsGroup"] = int(fs_group)
    if security_context:
        pod_spec["securityContext"] = security_context

    pod_labels = {"app": "a-evolve-trainer", "job-name": job_name}
    if labels:
        pod_labels.update(labels)

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": pod_labels,
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 24 * 3600,
            "template": {
                "metadata": {"labels": pod_labels},
                "spec": pod_spec,
            },
        },
    }
    return manifest


__all__ = ["build_job_manifest"]
