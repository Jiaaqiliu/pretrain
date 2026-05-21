"""Experiment launcher — submits and manages training jobs."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from autopilot.backends.base import ComputeBackend, JobConfig, JobHandle, JobStatus
from autopilot.experiment.config_builder import GeneratedConfig
from autopilot.utils.logging import get_logger
from autopilot.utils.persistence import ExperimentRecord, StateStore

log = get_logger("experiment.launcher")


class ExperimentLauncher:
    """Manages the lifecycle of training experiments.

    Responsibilities:
    - Convert GeneratedConfig → JobConfig → submit to backend
    - Track experiment state across submissions
    - Support batch/sweep submissions
    - Handle retries and resumption
    """

    def __init__(self, backend: ComputeBackend, store: StateStore):
        self._backend = backend
        self._store = store
        self._active_handles: Dict[str, JobHandle] = {}

    @property
    def active_experiments(self) -> List[str]:
        return list(self._active_handles.keys())

    def launch(self, config: GeneratedConfig) -> str:
        """Launch a single training experiment. Returns experiment_id."""
        experiment_id = self._generate_id()
        job_config = self._to_job_config(config)

        handle = self._backend.submit_job(job_config)
        self._active_handles[experiment_id] = handle

        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=config.name,
            config={
                "model": config.model_config,
                "optimizer": config.optimizer_config,
                "scheduler": config.scheduler_config,
                "data": config.data_config,
                "trainer": config.trainer_config,
                "launch": config.launch_config,
            },
            status="running",
        )
        self._store.save_experiment(record)

        log.info(f"Launched experiment {experiment_id} ({config.name}) → job {handle.job_id}")
        return experiment_id

    def launch_sweep(self, configs: List[GeneratedConfig]) -> List[str]:
        """Launch multiple experiments as a sweep."""
        experiment_ids = []
        for config in configs:
            eid = self.launch(config)
            experiment_ids.append(eid)
        log.info(f"Launched sweep of {len(experiment_ids)} experiments")
        return experiment_ids

    def cancel(self, experiment_id: str) -> None:
        """Cancel a running experiment."""
        handle = self._active_handles.get(experiment_id)
        if handle is None:
            log.warning(f"No active handle for experiment {experiment_id}")
            return

        self._backend.cancel_job(handle)
        self._update_status(experiment_id, "stopped")
        del self._active_handles[experiment_id]
        log.info(f"Cancelled experiment {experiment_id}")

    def cancel_all(self) -> None:
        """Cancel all active experiments."""
        for eid in list(self._active_handles.keys()):
            self.cancel(eid)

    def get_status(self, experiment_id: str) -> Optional[str]:
        """Get current status of an experiment."""
        handle = self._active_handles.get(experiment_id)
        if handle is None:
            record = self._store.get_experiment(experiment_id)
            return record.status if record else None

        status = self._backend.get_status(handle)
        status_str = status.value
        self._update_status(experiment_id, status_str)

        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            del self._active_handles[experiment_id]

        return status_str

    def get_handle(self, experiment_id: str) -> Optional[JobHandle]:
        """Get the job handle for an experiment."""
        return self._active_handles.get(experiment_id)

    def refresh_all_statuses(self) -> Dict[str, str]:
        """Refresh status of all active experiments."""
        statuses = {}
        for eid in list(self._active_handles.keys()):
            statuses[eid] = self.get_status(eid) or "unknown"
        return statuses

    def resume_from_checkpoint(
        self,
        experiment_id: str,
        checkpoint_path: str,
        config_overrides: Optional[Dict] = None,
    ) -> str:
        """Resume a failed/stopped experiment from a checkpoint with optional config changes."""
        record = self._store.get_experiment(experiment_id)
        if record is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        new_config = dict(record.config)
        if config_overrides:
            for key, value in config_overrides.items():
                parts = key.split(".")
                target = new_config
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value

        new_config.setdefault("trainer", {})["load_path"] = checkpoint_path

        # Create a new GeneratedConfig for the resumed run
        resumed_config = GeneratedConfig(
            name=f"{record.name}-resumed",
            model_config=new_config.get("model", {}),
            optimizer_config=new_config.get("optimizer", {}),
            scheduler_config=new_config.get("scheduler", {}),
            data_config=new_config.get("data", {}),
            trainer_config=new_config.get("trainer", {}),
            launch_config=new_config.get("launch", {}),
        )

        new_id = self.launch(resumed_config)
        log.info(f"Resumed experiment {experiment_id} → {new_id} from {checkpoint_path}")
        return new_id

    def _to_job_config(self, config: GeneratedConfig) -> JobConfig:
        """Convert a GeneratedConfig to a backend-agnostic JobConfig."""
        launch = config.launch_config

        # Build the training command
        command = self._build_training_command(config)

        env_vars = {
            "AUTOPILOT_EXPERIMENT_NAME": config.name,
            "WANDB_PROJECT": "autopilot",
            "WANDB_RUN_NAME": config.name,
        }

        return JobConfig(
            name=config.name,
            command=command,
            num_nodes=launch.get("num_nodes", 1),
            num_gpus_per_node=launch.get("gpus_per_node", 8),
            env_vars=env_vars,
            tags={"managed_by": "autopilot", "config_name": config.name},
        )

    def _build_training_command(self, config: GeneratedConfig) -> List[str]:
        """Build the torchrun command for training."""
        launch = config.launch_config
        overrides = config.to_olmo_overrides()

        cmd = [
            "torchrun",
            f"--nproc-per-node={launch.get('gpus_per_node', 8)}",
        ]

        if launch.get("num_nodes", 1) > 1:
            cmd.extend([
                f"--nnodes={launch['num_nodes']}",
                "--rdzv-backend=c10d",
            ])

        # The actual training script + overrides
        cmd.append("src/scripts/train/autopilot_run.py")
        cmd.append("train")
        cmd.append(config.name)
        cmd.extend(overrides)

        return cmd

    def _update_status(self, experiment_id: str, status: str) -> None:
        record = self._store.get_experiment(experiment_id)
        if record:
            record.status = status
            self._store.save_experiment(record)

    def _generate_id(self) -> str:
        return f"exp_{uuid.uuid4().hex[:12]}"
