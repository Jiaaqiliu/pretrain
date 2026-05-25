"""Autonomous Experiment Agent — self-healing GPU job orchestrator.

A persistent agent that runs on a CPU pod, managing GPU training jobs with
automatic failure diagnosis, recovery, and checkpoint resumption.

Architecture:
    ExperimentAgent (CPU pod, runs indefinitely)
        ├── JobSubmitter: creates and submits K8s Jobs
        ├── JobMonitor: polls job/pod status, detects failures
        ├── FailureDiagnoser: classifies failure root cause from logs
        ├── RecoveryStrategy: selects repair action per failure type
        ├── CheckpointManager: finds latest valid checkpoint for resume
        └── EventLog: records all decisions and outcomes

Failure taxonomy and recovery strategies:
    - NETWORK_DISCONNECT: resubmit same config (transient)
    - OOM: reduce microbatch_size or enable gradient checkpointing, resubmit
    - DATA_ERROR: check data paths, fix config, resubmit
    - TIMEOUT: extend timeout or reduce steps, resubmit
    - CODE_BUG: log error, alert human, do NOT retry
    - PREEMPTION: resubmit with checkpoint resume (node was reclaimed)
    - UNKNOWN: retry up to 3 times, then alert human

Usage:
    # Deploy as a long-running CPU pod on the cluster
    python -m agent_evolve.model.algorithms.autopretrain.experiment_agent \
        --config /fsx/dev/jiaqi/experiments/autopretrain/agent_config.yaml

    # Or run from the orchestrator
    agent = ExperimentAgent(config)
    agent.submit_experiment(trials)
    agent.run_until_complete()
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FailureType(Enum):
    NETWORK_DISCONNECT = "network_disconnect"
    OOM = "oom"
    DATA_ERROR = "data_error"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    TIMEOUT = "timeout"
    CODE_BUG = "code_bug"
    PREEMPTION = "preemption"
    PENDING_RESOURCES = "pending_resources"
    UNKNOWN = "unknown"


@dataclass
class RecoveryAction:
    action: str  # "resubmit", "resubmit_with_fix", "alert_human", "wait"
    description: str
    config_changes: dict[str, Any] = field(default_factory=dict)
    wait_seconds: int = 0


@dataclass
class JobEvent:
    timestamp: float
    job_name: str
    event_type: str  # "submitted", "running", "failed", "recovered", "completed"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialSpec:
    """Specification for a single training trial."""
    name: str
    trial_id: str
    config: dict[str, Any]  # Training config (mix, steps, model, etc.)
    max_retries: int = 5
    current_retry: int = 0
    checkpoint_path: str | None = None  # For resuming from failure


@dataclass
class AgentConfig:
    """Configuration for the Experiment Agent."""
    # K8s settings
    namespace: str = "default"
    context: str = "arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm"
    job_prefix: str = "luhanqin"
    node_selector: str = "trainer5"
    image: str = "801953956576.dkr.ecr.ap-south-1.amazonaws.com/ads-foundation-model-training/verl-multiturn:1.0.2"

    # Monitoring
    poll_interval: int = 60  # seconds between status checks
    max_retries_per_trial: int = 5
    timeout_per_trial: int = 14400  # 4 hours max

    # Paths
    code_dir: str = "/fsx/dev/jiaqi/A-EVOLVE-V2"
    checkpoint_base: str = "/fsx/dev/jiaqi/checkpoints/autopretrain-mvp"
    log_dir: str = "/fsx/dev/jiaqi/experiments/autopretrain/agent_logs"

    # Recovery
    oom_microbatch_reduction: float = 0.5  # halve microbatch on OOM
    network_retry_delay: int = 30  # seconds to wait before retry on network error


class FailureDiagnoser:
    """Diagnoses the root cause of a job failure from pod logs."""

    PATTERNS = {
        FailureType.OOM: [
            r"CUDA out of memory",
            r"OutOfMemoryError",
            r"torch\.cuda\.OutOfMemoryError",
        ],
        FailureType.NETWORK_DISCONNECT: [
            r"Connection closed by peer",
            r"NCCL WARN.*Connect",
            r"NCCL timeout",
            r"gloo.*Connection.*closed",
            r"Socket Timeout",
        ],
        FailureType.DATA_ERROR: [
            r"Token IDs.*outside valid range",
            r"FileNotFoundError",
            r"No data paths found",
            r"Pattern.*did not match any files",
        ],
        FailureType.FINGERPRINT_MISMATCH: [
            r"Dataset fingerprint does not match",
        ],
        FailureType.PREEMPTION: [
            r"node.*NotReady",
            r"Evicted",
            r"preempt",
        ],
        FailureType.TIMEOUT: [
            r"DeadlineExceeded",
            r"Job.*timed out",
        ],
        FailureType.PENDING_RESOURCES: [
            r"Insufficient nvidia\.com/gpu",
            r"Insufficient memory",
            r"Unschedulable",
        ],
    }

    def diagnose(self, pod_logs: str, pod_events: str = "") -> FailureType:
        """Classify failure type from logs and events."""
        combined = pod_logs + "\n" + pod_events

        for failure_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return failure_type

        # Check for Python exceptions that indicate code bugs
        if re.search(r"(ImportError|SyntaxError|NameError|AttributeError|TypeError)", combined):
            return FailureType.CODE_BUG

        return FailureType.UNKNOWN

    def get_error_context(self, pod_logs: str, max_lines: int = 20) -> str:
        """Extract relevant error context from logs."""
        lines = pod_logs.split("\n")
        error_lines = []
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ["error", "critical", "traceback", "exception"]):
                start = max(0, i - 2)
                end = min(len(lines), i + max_lines)
                error_lines.extend(lines[start:end])
                break
        return "\n".join(error_lines[-max_lines:])


class RecoveryStrategy:
    """Selects recovery actions based on failure diagnosis."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def get_action(
        self, failure_type: FailureType, trial: TrialSpec, retry_count: int
    ) -> RecoveryAction:
        if retry_count >= self.config.max_retries_per_trial:
            return RecoveryAction(
                action="alert_human",
                description=f"Max retries ({retry_count}) reached for {trial.name}",
            )

        strategies = {
            FailureType.NETWORK_DISCONNECT: self._handle_network,
            FailureType.OOM: self._handle_oom,
            FailureType.DATA_ERROR: self._handle_data_error,
            FailureType.FINGERPRINT_MISMATCH: self._handle_fingerprint,
            FailureType.TIMEOUT: self._handle_timeout,
            FailureType.PREEMPTION: self._handle_preemption,
            FailureType.PENDING_RESOURCES: self._handle_pending,
            FailureType.CODE_BUG: self._handle_code_bug,
            FailureType.UNKNOWN: self._handle_unknown,
        }

        handler = strategies.get(failure_type, self._handle_unknown)
        return handler(trial, retry_count)

    def _handle_network(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        return RecoveryAction(
            action="resubmit",
            description="Network disconnect (transient) — resubmit with checkpoint resume",
            wait_seconds=self.config.network_retry_delay,
        )

    def _handle_oom(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        current_microbatch = trial.config.get("rank_microbatch_size", 16384)
        new_microbatch = int(current_microbatch * self.config.oom_microbatch_reduction)
        return RecoveryAction(
            action="resubmit_with_fix",
            description=f"OOM — reduce microbatch {current_microbatch} → {new_microbatch}",
            config_changes={"rank_microbatch_size": new_microbatch},
        )

    def _handle_data_error(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        return RecoveryAction(
            action="alert_human",
            description="Data format/path error — requires manual fix",
        )

    def _handle_fingerprint(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        return RecoveryAction(
            action="resubmit_with_fix",
            description="Fingerprint mismatch — clear checkpoint dir and restart fresh",
            config_changes={"clear_checkpoint": True},
        )

    def _handle_timeout(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        return RecoveryAction(
            action="resubmit",
            description="Timeout — resubmit with checkpoint resume",
        )

    def _handle_preemption(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        return RecoveryAction(
            action="resubmit",
            description="Node preemption — resubmit with checkpoint resume",
            wait_seconds=60,
        )

    def _handle_pending(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        return RecoveryAction(
            action="wait",
            description="Resources unavailable — wait for GPU nodes to free up",
            wait_seconds=300,
        )

    def _handle_code_bug(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        return RecoveryAction(
            action="alert_human",
            description="Code bug detected — requires manual intervention",
        )

    def _handle_unknown(self, trial: TrialSpec, retry: int) -> RecoveryAction:
        if retry < 2:
            return RecoveryAction(
                action="resubmit",
                description=f"Unknown failure — retry {retry + 1}/{self.config.max_retries_per_trial}",
                wait_seconds=60,
            )
        return RecoveryAction(
            action="alert_human",
            description=f"Unknown failure persists after {retry} retries",
        )


class K8sClient:
    """Thin wrapper around kubectl for job management."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._ctx = ["--context", config.context]
        self._ns = ["-n", config.namespace]

    def _run(self, cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
        full_cmd = ["kubectl"] + cmd + self._ns + self._ctx
        return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)

    def submit_job(self, manifest: str) -> bool:
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"] + self._ns + self._ctx,
            input=manifest, capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0

    def delete_job(self, job_name: str) -> bool:
        result = self._run(["delete", "job", job_name, "--ignore-not-found"])
        return result.returncode == 0

    def get_job_status(self, job_name: str) -> str:
        """Returns: Running, Complete, Failed, Pending, NotFound"""
        result = self._run(["get", "job", job_name, "-o", "jsonpath={.status.conditions[-1:].type}"])
        status = result.stdout.strip()
        if not status:
            # Check if pods exist and are running
            pod_result = self._run(["get", "pods", "-l", f"job-name={job_name}", "-o", "jsonpath={.items[*].status.phase}"])
            phases = pod_result.stdout.strip().split()
            if "Running" in phases:
                return "Running"
            elif "Pending" in phases:
                return "Pending"
            elif not phases:
                return "NotFound"
            return "Running"  # Has pods but no terminal condition yet
        if status in ("Complete", "Failed"):
            return status
        return "Running"

    def get_pod_logs(self, job_name: str, tail: int = 100) -> str:
        result = self._run(["logs", f"job/{job_name}", "--tail", str(tail)])
        return result.stdout

    def get_pod_events(self, job_name: str) -> str:
        result = self._run(["get", "events", "--field-selector", f"involvedObject.name={job_name}"])
        return result.stdout

    def get_latest_checkpoint(self, save_folder: str) -> str | None:
        """Find the latest checkpoint step directory."""
        result = subprocess.run(
            ["kubectl", "exec", "jiaqi-omnimem-eval"] + self._ns + self._ctx +
            ["--", "bash", "-c", f"ls -d {save_folder}/step* 2>/dev/null | sort -V | tail -1"],
            capture_output=True, text=True, timeout=10,
        )
        path = result.stdout.strip()
        return path if path and "step0" not in path else None


class ExperimentAgent:
    """Autonomous experiment orchestrator with self-healing capabilities."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.k8s = K8sClient(config)
        self.diagnoser = FailureDiagnoser()
        self.recovery = RecoveryStrategy(config)
        self.events: list[JobEvent] = []
        self.trials: dict[str, TrialSpec] = {}

        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def submit_experiment(self, trials: list[TrialSpec]):
        """Submit a batch of trials and begin monitoring."""
        for trial in trials:
            self.trials[trial.trial_id] = trial
            self._submit_trial(trial)

    def run_until_complete(self, timeout: int | None = None):
        """Main loop: monitor, diagnose, recover, until all trials complete."""
        start = time.time()
        logger.info("Agent started. Monitoring %d trials.", len(self.trials))

        while True:
            if timeout and (time.time() - start) > timeout:
                logger.warning("Agent timeout reached.")
                break

            all_done = True
            for trial_id, trial in list(self.trials.items()):
                status = self._check_trial(trial)

                if status == "Complete":
                    self._log_event(trial.name, "completed", {"retry": trial.current_retry})
                    logger.info("✓ %s completed successfully.", trial.name)
                elif status == "Failed":
                    all_done = False
                    self._handle_failure(trial)
                elif status in ("Running", "Pending"):
                    all_done = False
                elif status == "alert_human":
                    logger.error("✗ %s requires human intervention.", trial.name)
                    # Don't retry, but don't block other trials
                    continue

            if all_done:
                logger.info("All trials complete!")
                break

            time.sleep(self.config.poll_interval)

        self._save_event_log()

    def _submit_trial(self, trial: TrialSpec):
        """Generate and submit a K8s Job for this trial."""
        job_name = f"{self.config.job_prefix}-{trial.trial_id}"

        # Delete any existing job with same name
        self.k8s.delete_job(job_name)
        time.sleep(2)

        # Build resume flag if checkpoint exists
        resume_args = ""
        if trial.checkpoint_path:
            resume_args = f" --save-folder {trial.checkpoint_path}"

        manifest = self._build_job_manifest(job_name, trial, resume_args)
        success = self.k8s.submit_job(manifest)

        if success:
            self._log_event(trial.name, "submitted", {"retry": trial.current_retry, "job": job_name})
            logger.info("Submitted %s (retry %d)", job_name, trial.current_retry)
        else:
            logger.error("Failed to submit %s", job_name)

    def _check_trial(self, trial: TrialSpec) -> str:
        """Check the current status of a trial's K8s job."""
        job_name = f"{self.config.job_prefix}-{trial.trial_id}"
        return self.k8s.get_job_status(job_name)

    def _handle_failure(self, trial: TrialSpec):
        """Diagnose failure and execute recovery."""
        job_name = f"{self.config.job_prefix}-{trial.trial_id}"
        logs = self.k8s.get_pod_logs(job_name, tail=200)
        events = self.k8s.get_pod_events(job_name)

        failure_type = self.diagnoser.diagnose(logs, events)
        error_context = self.diagnoser.get_error_context(logs)

        logger.warning(
            "Trial %s failed: %s\n  Context: %s",
            trial.name, failure_type.value, error_context[:200],
        )

        self._log_event(trial.name, "failed", {
            "failure_type": failure_type.value,
            "retry": trial.current_retry,
            "error_context": error_context[:500],
        })

        # Get recovery action
        action = self.recovery.get_action(failure_type, trial, trial.current_retry)
        logger.info("Recovery: %s", action.description)

        if action.action == "alert_human":
            self._log_event(trial.name, "alert_human", {"reason": action.description})
            # Mark as terminal — won't be retried
            trial.current_retry = self.config.max_retries_per_trial
            return

        if action.action == "wait":
            logger.info("Waiting %ds before retry...", action.wait_seconds)
            time.sleep(action.wait_seconds)

        # Apply config changes if needed
        if action.config_changes:
            trial.config.update(action.config_changes)
            if action.config_changes.get("clear_checkpoint"):
                trial.checkpoint_path = None

        # Check for checkpoint to resume from
        save_folder = f"{self.config.checkpoint_base}/{trial.name}"
        latest_ckpt = self.k8s.get_latest_checkpoint(save_folder)
        if latest_ckpt:
            trial.checkpoint_path = save_folder
            logger.info("Will resume from checkpoint: %s", latest_ckpt)

        # Wait before retry
        if action.wait_seconds:
            time.sleep(action.wait_seconds)

        # Increment retry and resubmit
        trial.current_retry += 1
        self._submit_trial(trial)
        self._log_event(trial.name, "recovered", {
            "action": action.action,
            "description": action.description,
            "retry": trial.current_retry,
        })

    def _build_job_manifest(self, job_name: str, trial: TrialSpec, extra_args: str = "") -> str:
        """Generate K8s Job YAML for a trial."""
        trial_arg = trial.config.get("trial_name", trial.name)
        microbatch = trial.config.get("rank_microbatch_size", 16384)

        return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {self.config.namespace}
  labels:
    experiment: autopretrain
    managed-by: experiment-agent
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      nodeSelector:
        eks.amazonaws.com/nodegroup: {self.config.node_selector}
      restartPolicy: Never
      containers:
      - name: worker
        image: {self.config.image}
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -uxo pipefail
          cd {self.config.code_dir}
          pip install -e olmo-core/.[all] > /dev/null 2>&1 || true
          set -e

          torchrun --nproc_per_node=8 --nnodes=1 \\
            --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \\
            experiments/autopretrain/mvp_3trial.py --trial {trial_arg}{extra_args}

          echo "Trial {trial_arg} DONE"
        env:
        - name: HF_HOME
          value: "/fsx/dev/jiaqi/.cache/huggingface"
        - name: CUDA_HOME
          value: "/opt/cuda-toolkit"
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "expandable_segments:True"
        resources:
          limits:
            nvidia.com/gpu: 8
          requests:
            cpu: "96"
            memory: "512Gi"
            nvidia.com/gpu: 8
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

    def _log_event(self, trial_name: str, event_type: str, details: dict = None):
        event = JobEvent(
            timestamp=time.time(),
            job_name=trial_name,
            event_type=event_type,
            details=details or {},
        )
        self.events.append(event)

    def _save_event_log(self):
        log_path = Path(self.config.log_dir) / "agent_events.jsonl"
        with open(log_path, "a") as f:
            for event in self.events:
                f.write(json.dumps({
                    "timestamp": event.timestamp,
                    "job": event.job_name,
                    "type": event.event_type,
                    "details": event.details,
                }) + "\n")
        logger.info("Event log saved to %s (%d events)", log_path, len(self.events))
        self.events.clear()


def main():
    """Entry point for running the Experiment Agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Experiment Agent")
    parser.add_argument("--trials", nargs="+", default=["llama3", "reasoning_heavy", "uniform"])
    parser.add_argument("--timeout", type=int, default=18000, help="Agent timeout in seconds")
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = AgentConfig(poll_interval=args.poll_interval)
    agent = ExperimentAgent(config)

    trials = [
        TrialSpec(
            name=name,
            trial_id=f"autopretrain-{name.replace('_', '-')}",
            config={"trial_name": name, "rank_microbatch_size": 16384},
        )
        for name in args.trials
    ]

    agent.submit_experiment(trials)
    agent.run_until_complete(timeout=args.timeout)


if __name__ == "__main__":
    main()
