"""Unit tests for ``job_manifest`` — manifest structure correctness."""

from __future__ import annotations

from agent_evolve.backends.tinkerlite.elastic.k8s.job_manifest import build_job_manifest


def _default_args(**overrides):
    defaults = dict(
        job_name="test-job",
        namespace="a-evolve",
        image="a-evolve/trainer:latest",
        cfg_path="/fsx/zzsamshi/a-evolve/runs/foo/.ddp_config.json",
        world_size=8,
        pvc_name="fsx-zzsamshi",
    )
    defaults.update(overrides)
    return defaults


def test_basic_structure() -> None:
    m = build_job_manifest(**_default_args())
    assert m["apiVersion"] == "batch/v1"
    assert m["kind"] == "Job"
    assert m["metadata"]["name"] == "test-job"
    assert m["metadata"]["namespace"] == "a-evolve"
    assert m["spec"]["backoffLimit"] == 0


def test_command_uses_torchrun_module() -> None:
    m = build_job_manifest(**_default_args(world_size=4))
    cmd = m["spec"]["template"]["spec"]["containers"][0]["command"]
    # Verify the worker entrypoint matches the single-node DDP launcher
    # exactly. A divergence here would silently produce different results on
    # k8s vs. local.
    assert "-m" in cmd and "torch.distributed.run" in cmd
    assert "--nproc_per_node=4" in cmd
    assert "-m" in cmd and "agent_evolve.training.runners.ddp_worker" in cmd
    assert "--config" in cmd
    assert cmd[cmd.index("--config") + 1] == "/fsx/zzsamshi/a-evolve/runs/foo/.ddp_config.json"


def test_gpu_resource_limit_matches_world_size() -> None:
    m = build_job_manifest(**_default_args(world_size=8))
    resources = m["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert resources["limits"]["nvidia.com/gpu"] == "8"
    assert resources["requests"]["nvidia.com/gpu"] == "8"


def test_fsx_pvc_mount() -> None:
    m = build_job_manifest(**_default_args(pvc_name="my-pvc", pvc_mount_path="/shared"))
    pod = m["spec"]["template"]["spec"]
    vols = {v["name"]: v for v in pod["volumes"]}
    assert vols["fsx"]["persistentVolumeClaim"]["claimName"] == "my-pvc"

    mounts = m["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    mount_by_name = {m["name"]: m for m in mounts}
    assert mount_by_name["fsx"]["mountPath"] == "/shared"


def test_node_selector_applied() -> None:
    m = build_job_manifest(
        **_default_args(node_selector={"nvidia.com/gpu.product": "H200"})
    )
    assert m["spec"]["template"]["spec"]["nodeSelector"] == {
        "nvidia.com/gpu.product": "H200"
    }


def test_no_node_selector_when_none() -> None:
    m = build_job_manifest(**_default_args(node_selector=None))
    assert "nodeSelector" not in m["spec"]["template"]["spec"]


def test_env_includes_offline_flags() -> None:
    m = build_job_manifest(**_default_args())
    env = {e["name"]: e["value"] for e in m["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["WANDB_DISABLED"] == "true"
    assert env["PYTHONPATH"] == "/fsx/zzsamshi/a-evolve"


def test_restart_policy_never() -> None:
    m = build_job_manifest(**_default_args())
    assert m["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def test_shared_memory_volume() -> None:
    m = build_job_manifest(**_default_args(shm_size_gib=32))
    vols = {v["name"]: v for v in m["spec"]["template"]["spec"]["volumes"]}
    assert vols["dshm"]["emptyDir"]["sizeLimit"] == "32Gi"
    assert vols["dshm"]["emptyDir"]["medium"] == "Memory"
