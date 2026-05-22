#!/usr/bin/env python3
"""Generate K8s PyTorchJob YAML files for all proxy training experiments.

Generates: 4 schedules × 2 model sizes × 3 seeds = 24 YAML files

Usage:
    python scripts/k8s/thermo/gen_training_jobs.py
"""

from pathlib import Path

SCHEDULES = ["cosine", "wsd_linear", "wsd_exponential", "gaussian"]
SEEDS = [42, 123, 456]

CONFIGS = {
    "190m": {
        "model_size": "190M",
        "nodes": 1,
        "workers": 0,  # 1 master only
        "max_steps": 25000,
        "memory": "512Gi",
    },
    "1b": {
        "model_size": "1B",
        "nodes": 2,
        "workers": 1,  # 1 master + 1 worker
        "max_steps": 50000,
        "memory": "1800Gi",
    },
}


def gen_yaml(model_key: str, schedule: str, seed: int) -> str:
    cfg = CONFIGS[model_key]
    job_name = f"jiaqi-thermo-{model_key}-{schedule.replace('_', '-')}-s{seed}"
    output_dir = f"/fsx/dev/jiaqi/thermo_experiments/{model_key}_{schedule}_s{seed}"

    nnodes = cfg["nodes"]
    train_cmd = (
        f"torchrun \\\n"
        f"                --nproc_per_node=$GPUS_PER_NODE \\\n"
        f"                --nnodes={nnodes} \\\n"
        f"                --rdzv_backend=c10d \\\n"
        f"                --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} \\\n"
        f"                /fsx/dev/jiaqi/A-EVOLVE-V2/scripts/thermo/train_schedule_comparison.py \\\n"
        f"                --model-size {cfg['model_size']} \\\n"
        f"                --schedule {schedule} \\\n"
        f"                --seed {seed} \\\n"
        f"                --max-steps {cfg['max_steps']} \\\n"
        f"                --output-dir {output_dir} \\\n"
        f"                --measure-interval 200 \\\n"
        f"                --checkpoint-interval 200 \\\n"
        f"                --wandb-project thermo-pretraining"
    )

    env_block = """            env:
            - name: WANDB_PROJECT
              value: "thermo-pretraining"
            - name: WANDB_RUN_ID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  key: WANDB_API_KEY
                  name: wandb-secret-jiaqi
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_DEBUG_SUBSYS
              value: "NET"
            - name: FI_PROVIDER
              value: "efa"
            - name: FI_EFA_USE_DEVICE_RDMA
              value: "1"
            - name: RDMAV_FORK_SAFE
              value: "1"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: NCCL_TIMEOUT
              value: "600"
            - name: CUDA_HOME
              value: "/opt/cuda-toolkit"
            - name: GPUS_PER_NODE
              value: "8"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True" """

    resources = f"""            resources:
              limits:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 32
              requests:
                cpu: "96"
                memory: "{cfg['memory']}"
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 32"""

    volumes = """            volumeMounts:
            - mountPath: /fsx
              name: fsx
            - mountPath: /dev/shm
              name: dshm
          volumes:
          - name: fsx
            persistentVolumeClaim:
              claimName: fsx
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: "1000Gi" """

    container_block = f"""          containers:
          - name: pytorch
            image: 801953956576.dkr.ecr.ap-south-1.amazonaws.com/ads-foundation-model-training/verl-multiturn:1.0.2
            imagePullPolicy: Always
            command: ["/bin/bash", "-c"]
            args:
            - |
              set -euxo pipefail
              echo "======== Thermo Training: {cfg['model_size']} / {schedule} / seed={seed} ========"

              pip install -e /fsx/dev/jiaqi/A-EVOLVE-V2/olmo-core/.[all] > /tmp/pip_olmo.log 2>&1 || true
              tail -5 /tmp/pip_olmo.log
              pip install scipy matplotlib > /tmp/pip_deps.log 2>&1 || true
              tail -3 /tmp/pip_deps.log

              {train_cmd}

              exitcode=$?
              if [ $exitcode -ne 0 ]; then
                echo "Job failed (exit code: $exitcode). Sleeping 3600s for debugging..."
                sleep 3600
              fi
              exit 0
{env_block}
{resources}
{volumes}"""

    # Master spec
    master_spec = f"""    Master:
      replicas: 1
      template:
        metadata:
          labels:
            role: master
            job: {job_name}
        spec:
          nodeSelector:
            eks.amazonaws.com/nodegroup: trainer5
          restartPolicy: Never
{container_block}"""

    # Worker spec (only for multi-node)
    worker_spec = ""
    if cfg["workers"] > 0:
        # Worker uses same container but without WandB env vars
        worker_container = f"""          containers:
          - name: pytorch
            image: 801953956576.dkr.ecr.ap-south-1.amazonaws.com/ads-foundation-model-training/verl-multiturn:1.0.2
            imagePullPolicy: Always
            command: ["/bin/bash", "-c"]
            args:
            - |
              set -euxo pipefail
              pip install -e /fsx/dev/jiaqi/A-EVOLVE-V2/olmo-core/.[all] > /tmp/pip_olmo.log 2>&1 || true
              tail -5 /tmp/pip_olmo.log
              pip install scipy matplotlib > /tmp/pip_deps.log 2>&1 || true
              tail -3 /tmp/pip_deps.log

              {train_cmd}

              exitcode=$?
              if [ $exitcode -ne 0 ]; then
                echo "Job failed. Sleeping 3600s for debugging..."
                sleep 3600
              fi
              exit 0
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_DEBUG_SUBSYS
              value: "NET"
            - name: FI_PROVIDER
              value: "efa"
            - name: FI_EFA_USE_DEVICE_RDMA
              value: "1"
            - name: RDMAV_FORK_SAFE
              value: "1"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: NCCL_TIMEOUT
              value: "600"
            - name: CUDA_HOME
              value: "/opt/cuda-toolkit"
            - name: GPUS_PER_NODE
              value: "8"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True"
{resources}
{volumes}"""

        worker_spec = f"""
    Worker:
      replicas: {cfg['workers']}
      template:
        metadata:
          labels:
            role: worker
            job: {job_name}
        spec:
          nodeSelector:
            eks.amazonaws.com/nodegroup: trainer5
          restartPolicy: Never
{worker_container}"""

    yaml = f"""# PyTorchJob: Thermo proxy training — {cfg['model_size']} / {schedule} / seed={seed}
# 节点: {nnodes} × 8 GPUs = {nnodes * 8} GPUs total
# 预计耗时: {'~32h (256 H100-hours)' if model_key == '190m' else '~128h (2048 H100-hours)'}
#
# kubectl apply -f {Path(job_name).name}.yaml -n default \\
#   --context arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm
#
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {job_name}
  namespace: default
spec:
  runPolicy:
    backoffLimit: 0
    cleanPodPolicy: None
    ttlSecondsAfterFinished: 7200
  pytorchReplicaSpecs:
{master_spec}{worker_spec}
"""
    return yaml


def main():
    output_dir = Path(__file__).parent
    generated = []

    for model_key in CONFIGS:
        for schedule in SCHEDULES:
            for seed in SEEDS:
                yaml = gen_yaml(model_key, schedule, seed)
                filename = f"thermo_train_{model_key}_{schedule}_s{seed}.yaml"
                filepath = output_dir / filename
                filepath.write_text(yaml)
                generated.append(filename)

    print(f"Generated {len(generated)} K8s job files:")
    for f in sorted(generated):
        print(f"  {f}")


if __name__ == "__main__":
    main()
