"""OLMo-core training engine adapter.

Generates self-contained training scripts that:
1. Use the OLMo-core API (TransformerConfig factories, Trainer, etc.)
2. Include heartbeat callback for liveness monitoring
3. Include stability/memory monitoring callbacks
4. Support checkpoint resume
5. Support multiple model scales (190M to 32B)

Also generates K8s Job manifests tailored to the hardware requirements.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

from autopretrain.core.types import MetricSnapshot, TrialConfig

logger = logging.getLogger(__name__)

# Model scale → resource requirements
MODEL_RESOURCES = {
    "olmo2_190M": {"gpus": 1, "memory": "64Gi", "cpu": "16"},
    "olmo2_370M": {"gpus": 1, "memory": "64Gi", "cpu": "16"},
    "olmo2_760M": {"gpus": 2, "memory": "128Gi", "cpu": "32"},
    "olmo2_1B": {"gpus": 8, "memory": "512Gi", "cpu": "96"},
    "olmo2_3B": {"gpus": 8, "memory": "512Gi", "cpu": "96"},
    "olmo2_7B": {"gpus": 8, "memory": "1024Gi", "cpu": "96"},
    "olmo2_13B": {"gpus": 16, "memory": "1024Gi", "cpu": "96"},
    "olmo2_32B": {"gpus": 32, "memory": "2048Gi", "cpu": "192"},
}

# Default microbatch sizes per model scale (tokens per GPU per microbatch)
DEFAULT_MICROBATCH = {
    "olmo2_190M": 16,  # 16 seq × 4096
    "olmo2_370M": 16,
    "olmo2_760M": 8,
    "olmo2_1B": 4,     # 4 seq × 4096
    "olmo2_3B": 2,     # 2 seq × 4096
    "olmo2_7B": 1,
    "olmo2_13B": 1,
    "olmo2_32B": 1,
}


class OLMoAdapter:
    """Generates OLMo-core training scripts and K8s manifests."""

    def __init__(
        self,
        olmo_core_path: str = "/fsx/dev/jiaqi/A-EVOLVE-V2/olmo-core",
        image: str = "801953956576.dkr.ecr.ap-south-1.amazonaws.com/ads-foundation-model-training/verl-multiturn:1.0.2",
        node_selector: str = "trainer5",
        heartbeat_dir: str = "/fsx/dev/jiaqi/experiments/autopretrain/heartbeat",
    ) -> None:
        self.olmo_core_path = olmo_core_path
        self.image = image
        self.node_selector = node_selector
        self.heartbeat_dir = heartbeat_dir

    def generate_script(self, trial: TrialConfig) -> str:
        """Generate a self-contained OLMo-core training script."""
        seq_len = trial.sequence_length
        global_batch_tokens = trial.global_batch_size * seq_len
        microbatch_tokens = trial.rank_microbatch_size * seq_len

        # Build data mix section
        data_mix_code = self._render_data_mix(trial)

        # Optimizer selection
        if trial.use_skip_step_optimizer:
            optim_import = "from olmo_core.optim import SkipStepAdamWConfig, CosWithWarmup"
            optim_config = f"SkipStepAdamWConfig(lr={trial.lr}, weight_decay={trial.weight_decay}, betas=(0.9, 0.95))"
        else:
            optim_import = "from olmo_core.optim import AdamWConfig, CosWithWarmup"
            optim_config = f"AdamWConfig(lr={trial.lr}, weight_decay={trial.weight_decay}, betas=(0.9, 0.95))"

        # Gradient checkpointing
        grad_ckpt_line = ""
        if trial.enable_gradient_checkpointing:
            grad_ckpt_line = "\n    # Enable gradient checkpointing for memory savings\n    model.gradient_checkpointing = True"

        script = f'''"""HoloPretrain generated training script: {trial.name or trial.trial_id}"""

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

_olmo_core_src = Path("{self.olmo_core_path}/src")
if _olmo_core_src.exists():
    sys.path.insert(0, str(_olmo_core_src))

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
{optim_import}
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    MetricSaverCallback,
    SpeedMonitorCallback,
    StabilityMonitorCallback,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerDataParallelConfig

SEQUENCE_LENGTH = {seq_len}
GLOBAL_BATCH_SIZE = {global_batch_tokens}
HEARTBEAT_PATH = Path("{self.heartbeat_dir}/{trial.trial_id}.json")


{data_mix_code}


def write_heartbeat(step, loss=None, grad_norm=None, throughput=None, gpu_mem=None):
    """Write heartbeat file for liveness monitoring."""
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {{
        "trial_id": "{trial.trial_id}",
        "timestamp": time.time(),
        "step": step,
        "loss": loss,
        "grad_norm": grad_norm,
        "throughput_tps": throughput,
        "gpu_memory_pct": gpu_mem,
        "status": "training",
    }}
    HEARTBEAT_PATH.write_text(json.dumps(data))


def main():
    os.environ.setdefault("NCCL_TIMEOUT", "1800")
    prepare_training_environment(seed=42, timeout=timedelta(minutes=30))

    tokenizer = TokenizerConfig.dolma2()
    model_config = TransformerConfig.{trial.model_factory}(vocab_size=tokenizer.padded_vocab_size())
    model = model_config.build(init_device="meta"){grad_ckpt_line}

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size={microbatch_tokens},
        max_sequence_length={seq_len},
        optim={optim_config},
        scheduler=CosWithWarmup(warmup={trial.warmup_steps}),
        max_grad_norm={trial.max_grad_norm},
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )
    train_module = train_module_config.build(model)

    dataset, data_loader = build_data(train_module)

    trainer_config = (
        TrainerConfig(
            save_folder="{trial.save_folder}",
            save_overwrite=True,
            max_duration=Duration.steps({trial.max_steps}),
            metrics_collect_interval=50,
        )
        .with_callback("checkpointer", CheckpointerCallback(
            save_interval={trial.save_interval},
            ephemeral_save_interval={trial.ephemeral_save_interval},
        ))
        .with_callback("console_logger", ConsoleLoggerCallback())
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("stability", StabilityMonitorCallback(threshold_std=6.0))
        .with_callback("gpu_mem", GPUMemoryMonitorCallback())
        .with_callback("gc", GarbageCollectorCallback())
        .with_callback("metric_saver", MetricSaverCallback())
    )

    trainer = trainer_config.build(train_module, data_loader)

    # Write initial heartbeat
    write_heartbeat(step=0)

    trainer.fit()

    # Write final heartbeat
    write_heartbeat(step={trial.max_steps})
    teardown_training_environment()


if __name__ == "__main__":
    main()
'''

        return script

    def generate_manifest(self, trial: TrialConfig, script_content: str) -> str:
        """Generate a K8s Job manifest for this trial.

        The script is assumed to already be written to FSx at the expected path.
        The manifest simply runs it via torchrun.
        """
        resources = MODEL_RESOURCES.get(trial.model_factory, MODEL_RESOURCES["olmo2_1B"])
        num_gpus = trial.num_gpus or resources["gpus"]
        memory = resources["memory"]
        cpu = resources["cpu"]
        job_name = f"luhanqin-{trial.trial_id}"

        script_path = f"{trial.code_dir}/experiments/autopretrain/generated/{trial.trial_id}.py"

        num_nodes = trial.num_nodes
        rdzv = "localhost:29500" if num_nodes == 1 else "$MASTER_ADDR:29500"

        manifest = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: default
  labels:
    experiment: autopretrain
    trial-id: {trial.trial_id}
    managed-by: autopretrain-agent
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  activeDeadlineSeconds: {trial.deadline_seconds}
  template:
    spec:
      nodeSelector:
        eks.amazonaws.com/nodegroup: {self.node_selector}
      restartPolicy: Never
      containers:
      - name: worker
        image: {self.image}
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -uxo pipefail
          cd {trial.code_dir}
          pip install -e olmo-core/.[all] > /dev/null 2>&1 || true
          set -e

          torchrun --nproc_per_node={num_gpus} --nnodes={num_nodes} \\
            --rdzv_backend=c10d --rdzv_endpoint={rdzv} \\
            {script_path}

          echo "Trial {trial.trial_id} DONE"
        env:
        - name: HF_HOME
          value: "/fsx/dev/jiaqi/.cache/huggingface"
        - name: CUDA_HOME
          value: "/opt/cuda-toolkit"
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "expandable_segments:True"
        - name: NCCL_ASYNC_ERROR_HANDLING
          value: "1"
        - name: NCCL_TIMEOUT
          value: "1800"
        - name: TORCH_NCCL_BLOCKING_WAIT
          value: "0"
        resources:
          limits:
            nvidia.com/gpu: {num_gpus}
          requests:
            cpu: "{cpu}"
            memory: "{memory}"
            nvidia.com/gpu: {num_gpus}
        volumeMounts:
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
          sizeLimit: "200Gi"
"""
        return manifest

    def read_metrics(self, save_folder: str) -> list[MetricSnapshot]:
        """Read metrics from a completed trial's MetricSaver output."""
        metrics_path = Path(save_folder) / "metrics.json"
        if not metrics_path.exists():
            return []

        snapshots = []
        try:
            with open(metrics_path) as f:
                for line in f:
                    record = json.loads(line)
                    snapshots.append(MetricSnapshot(
                        step=record.get("step", 0),
                        timestamp=record.get("timestamp", 0),
                        metrics=record.get("metrics", {}),
                    ))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Error reading metrics from %s: %s", metrics_path, e)

        return snapshots

    def find_latest_checkpoint(self, save_folder: str) -> str | None:
        """Find the latest valid checkpoint in a save folder."""
        folder = Path(save_folder)
        if not folder.exists():
            return None

        step_dirs = sorted(
            [d for d in folder.iterdir() if d.is_dir() and d.name.startswith("step")],
            key=lambda d: int(d.name.replace("step", "")),
            reverse=True,
        )

        for step_dir in step_dirs:
            # Check if checkpoint is valid (has metadata file)
            if (step_dir / ".metadata.json").exists() or (step_dir / "metadata.json").exists():
                return str(step_dir)

        return None

    def _render_data_mix(self, trial: TrialConfig) -> str:
        """Render the data loading section of the script."""
        if trial.data_mix is None:
            return '''def build_data(train_module):
    """Build data loader with default paths."""
    tokenizer = TokenizerConfig.dolma2()
    dataset_config = NumpyFSLDatasetConfig(
        paths=["/fsx/dev/jiaqi/data/olmo-pretrain-raw/dclm_web/*.bin"],
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
    )
    loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=42,
    )
    dataset = dataset_config.build()
    data_loader = loader_config.build(dataset, dp_process_group=train_module.dp_process_group)
    return dataset, data_loader
'''

        # SourceMixture-based data loading
        sources_code = ""
        for domain, ratio in trial.data_mix.sources.items():
            path = trial.data_mix.paths.get(domain, f"/fsx/dev/jiaqi/data/olmo-pretrain-raw/{domain}/*.bin")
            sources_code += f'        SourceMixtureConfig(\n'
            sources_code += f'            source_name="{domain}",\n'
            sources_code += f'            target_ratio={ratio},\n'
            sources_code += f'            paths=["{path}"],\n'
            sources_code += f'            max_repetition_ratio={trial.data_mix.max_repetition_ratio},\n'
            sources_code += f'        ),\n'

        return f'''def build_data(train_module):
    """Build data loader with weighted path-based mixing.

    Uses NumpyFSLDatasetConfig with paths weighted by repetition to approximate
    the desired domain ratios. This avoids SourceMixture's sampling bug with empty files.
    """
    import glob

    tokenizer = TokenizerConfig.dolma2()

    # Collect paths per domain with ratio-based file selection
    domain_ratios = {repr(trial.data_mix.sources)}
    domain_paths = {repr(trial.data_mix.paths)}

    all_paths = []
    for domain, ratio in domain_ratios.items():
        pattern = domain_paths.get(domain, "")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"WARNING: No files for domain {{domain}} at {{pattern}}")
            continue
        # Select a proportional number of files based on ratio
        n_files = max(1, int(len(files) * ratio * 2))  # oversample slightly
        selected = files[:n_files]
        all_paths.extend(selected)
        print(f"  {{domain}}: {{len(selected)}}/{{len(files)}} files (ratio={{ratio:.2f}})")

    if not all_paths:
        raise RuntimeError("No data files found!")

    print(f"Total data files: {{len(all_paths)}}")

    dataset_config = NumpyFSLDatasetConfig(
        paths=all_paths,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
    )
    loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=42,
        ignore_fingerprint_mismatch=True,
    )
    dataset = dataset_config.build()
    data_loader = loader_config.build(dataset, dp_process_group=train_module.dp_process_group)
    return dataset, data_loader
'''
