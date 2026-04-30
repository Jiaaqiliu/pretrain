# K8s Elastic Backend (`k8s_h200`)

K8s-first, local-fallback backend for `TrainingEvolver`. Runs DDP training
stages on a shared Kubernetes cluster when capacity is available, falls
back to the local machine (if enabled) when cluster capacity is exhausted
or queue wait exceeds the threshold.

## Positioning

| Registry key | Where stages run | Use case |
|---|---|---|
| `h200_single_node` | Always local, single process | Development / single deep trial / zero infra dependency |
| `k8s_h200` | K8s cluster first → local fallback | Batch / sweep / production; absorbs cluster contention |

`k8s_h200` inherits `h200_single_node`'s pipeline orchestration and only
swaps the stage-spawn boundary. Trial pipeline semantics (SFT → RL → eval)
are identical.

## Prerequisites

1. **FSx PVC** named `fsx-zzsamshi` (default; override via
   `pvc_name=...`) that mounts the same FSx filesystem accessible on the
   host at `/fsx`. Inside pods it mounts at `pvc_mount_path=/fsx`.
2. **Node labels** so the scheduler can target H200 nodes. Default
   `node_selector` is None (no restriction) — pass
   `node_selector={"nvidia.com/gpu.product": "H200"}` or similar.
3. **Container image** built from [`Dockerfile`](./Dockerfile) and
   pushed to a registry the cluster can pull from. Default image name
   `a-evolve/trainer:latest` — override via `image=...`.
4. **kubeconfig** readable by the driver process. Either `$KUBECONFIG`,
   `~/.kube/config`, or pass `kubeconfig="/path/to/config"` explicitly.
   In-cluster service accounts (`load_incluster_config`) are also supported.
5. **`kubernetes` Python package**: `pip install "kubernetes>=31.0"` or
   install the repo with the `k8s` extra: `pip install -e '.[k8s]'`.

## Quick start

```python
from agent_evolve.training.api import TrainingEvolver

evolver = TrainingEvolver(
    workspace="seed_workspaces/nemotron_reasoner",
    benchmark="nemo_reasoner",
    algorithm="mcgs",
    backend="k8s_h200",            # instead of "h200_single_node"
)
evolver.run(cycles=4)
```

MCGS drives the trials exactly as before; the backend submits each DDP
stage to the elastic scheduler.

### Construction options

All configurable via `resolve_backend` kwargs (or by instantiating the
class directly):

| Kwarg | Default | Meaning |
|---|---|---|
| `namespace` | `"a-evolve"` | K8s namespace to create Jobs in |
| `image` | `"a-evolve/trainer:latest"` | Image for the `trainer` container |
| `pvc_name` | `"fsx-zzsamshi"` | PersistentVolumeClaim to mount at `/fsx` |
| `node_selector` | None | Label selector for H200 nodes |
| `queue_timeout_secs` | 600 | How long to tolerate `Pending` on k8s before falling back |
| `local_enabled` | True | Set False once the local H200 box is retired |
| `local_gpu_pool` | (0..7) | Physical GPU ids available locally |

## Scheduling policy

Per stage:

1. Ask each target (k8s, local) for a capacity probe.
2. If any target reports `can_run_now`, submit + block.
3. Else if k8s reports `can_queue`, submit there, tolerate `Pending` for
   `queue_timeout_secs`, then fall back on `PendingTimeout`.
4. Else try remaining targets.
5. Else raise `CapacityExhausted` (caller — MCGS — treats as trial failure).

Once a Job is `Running`, no stage-level timeout is imposed — trial budgets
are managed at the MCGS level via `TrialBudget`.

## Parallel LR sweep (fan-out)

See [examples/nemo_reasoning_example/drive_k8s_lr_sweep.py](../../../../examples/nemo_reasoning_example/drive_k8s_lr_sweep.py).
The backend exposes `submit_stage_async` / `wait_any` / `cancel_stage`
for callers that want to dispatch multiple trials concurrently.

## Debugging

### Inspect a manifest without submitting

```python
from agent_evolve.backends.tinkerlite.elastic.k8s.job_manifest import build_job_manifest
import yaml, json
m = build_job_manifest(
    job_name="test-job", namespace="a-evolve",
    image="a-evolve/trainer:latest",
    cfg_path="/fsx/zzsamshi/a-evolve/runs/foo/.ddp_config.json",
    world_size=8, pvc_name="fsx-zzsamshi",
)
print(yaml.dump(m))
```

Pipe into `kubectl apply -f -` to submit manually.

### Logs

Pod stdout/stderr is tailed to `<workspace>/logs/k8s_stages/<stage>.k8s.log`
while the Job runs. After the Job finishes, use `kubectl logs job/<name>`
for the authoritative view.

### Force local-only

Pass `local_enabled=True` and rely on `K8sComputeTarget` erroring out
(e.g. no `kubernetes` package installed) to short-circuit to local. Or
explicitly construct with a test kubeconfig pointing at an empty cluster.

## File layout

- [`backend.py`](backend.py) — `K8sTinkerLiteBackend` (extends `SingleNodeTinkerLiteBackend`)
- [`scheduler.py`](scheduler.py) — `ElasticScheduler`: priority-ordered probe + queue logic
- [`compute_target.py`](compute_target.py) — `ComputeTarget` Protocol, `CapacityReport`
- [`k8s_target.py`](k8s_target.py) — K8s Job submission, poll, log tail
- [`local_target.py`](local_target.py) — torchrun subprocess + GPU lock integration
- [`gpu_lock.py`](gpu_lock.py) — `flock`-based GPU reservation with stale PID cleanup
- [`job_manifest.py`](job_manifest.py) — `batch/v1 Job` manifest builder
- [`Dockerfile`](Dockerfile) — thin trainer image (code mounted, not baked)
