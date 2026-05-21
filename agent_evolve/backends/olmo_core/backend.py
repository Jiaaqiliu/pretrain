"""OLMo-core backend implementing the TrainingJobRunner protocol.

This backend enables the MCGS evolution loop to run pre-training and
fine-tuning experiments using OLMo-core's training infrastructure. It:

1. Reads workspace YAML configs (model/base.yaml, train/optimizer.yaml, etc.)
2. Translates them to OLMo-core TrainerConfig / TransformerConfig
3. Generates a self-contained training script
4. Launches the script via torchrun (local) or SLURM
5. Monitors training via metric files (JSONL)
6. Returns metrics + checkpoint path as TrainingTrialResult

Implements: agent_evolve.model.runner_protocol.TrainingJobRunner
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from agent_evolve.backends.olmo_core.config_translator import (
    OLMoCoreConfigTranslator,
    OLMoCoreTrainingConfig,
)
from agent_evolve.backends.olmo_core.script_generator import OLMoCoreScriptGenerator
from agent_evolve.model.types import (
    CheckpointRef,
    EvalMetrics,
    ErrorBuckets,
    TrainingSearchNode,
    TrainingTrialResult,
    TrialBudget,
    ValidityReport,
)

logger = logging.getLogger(__name__)


class OLMoCoreBackend:
    """Implements TrainingJobRunner using OLMo-core as the training engine."""

    name = "olmo_core"

    def __init__(
        self,
        *,
        mock: bool = False,
        gpus_per_node: int = 8,
        num_nodes: int = 1,
        use_slurm: bool = False,
        slurm_partition: str | None = None,
        slurm_account: str | None = None,
        olmo_core_path: str | None = None,
    ) -> None:
        self.mock = mock
        self.gpus_per_node = gpus_per_node
        self.num_nodes = num_nodes
        self.use_slurm = use_slurm
        self.slurm_partition = slurm_partition
        self.slurm_account = slurm_account

        if olmo_core_path:
            self._olmo_core_path = Path(olmo_core_path)
        else:
            self._olmo_core_path = Path(__file__).resolve().parents[3] / "olmo-core"

        self._translator = OLMoCoreConfigTranslator()
        self._generator = OLMoCoreScriptGenerator(olmo_core_path=self._olmo_core_path)

    def run_trial(
        self,
        workspace: Any,
        node: TrainingSearchNode,
        budget: TrialBudget,
        benchmark: Any,
    ) -> TrainingTrialResult:
        """Run a single training trial using OLMo-core.

        Flow:
            workspace YAML → OLMoCoreTrainingConfig → script → launch → monitor → result
        """
        t0 = time.time()
        workspace_path = str(workspace.root)
        trial_dir = Path(workspace.root) / "trials" / node.node_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        try:
            training_config = self._translator.translate(Path(workspace.root))
        except Exception as exc:
            logger.exception("Config translation failed for node %s", node.node_id)
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=workspace_path,
                status="train_failed",
                train_metrics={"error": f"config_translation: {exc}"},
                cost={"seconds": time.time() - t0},
            )

        training_config.save_folder = str(trial_dir / "checkpoints")
        training_config.metric_report_url = str(trial_dir / "metrics")
        # Scale global_batch_size proportionally when overriding node count
        old_total_gpus = training_config.num_nodes * training_config.gpus_per_node
        training_config.num_nodes = self.num_nodes
        training_config.gpus_per_node = self.gpus_per_node
        new_total_gpus = self.num_nodes * self.gpus_per_node
        if old_total_gpus > 0 and new_total_gpus != old_total_gpus:
            training_config.global_batch_size = int(
                training_config.global_batch_size * new_total_gpus / old_total_gpus
            )

        if budget.steps:
            training_config.max_steps = min(training_config.max_steps, budget.steps)

        if self.mock:
            return self._run_mock_trial(
                workspace, node, training_config, trial_dir, t0
            )

        try:
            script_path = self._generator.generate(training_config, trial_dir)
        except Exception as exc:
            logger.exception("Script generation failed for node %s", node.node_id)
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=workspace_path,
                status="train_failed",
                train_metrics={"error": f"script_generation: {exc}"},
                cost={"seconds": time.time() - t0},
            )

        try:
            train_metrics = self._launch_and_wait(
                script_path, training_config, budget, trial_dir
            )
        except Exception as exc:
            logger.exception("Training launch failed for node %s", node.node_id)
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=workspace_path,
                status="train_failed",
                train_metrics={"error": f"launch_failed: {exc}"},
                cost={"seconds": time.time() - t0},
            )

        checkpoint = self._find_latest_checkpoint(training_config.save_folder)
        if checkpoint is None:
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=workspace_path,
                status="train_failed",
                train_metrics=train_metrics,
                cost={"seconds": time.time() - t0},
            )

        try:
            eval_metrics, error_buckets = self._run_evaluation(
                workspace, checkpoint, benchmark
            )
        except Exception as exc:
            logger.exception("Evaluation failed for node %s", node.node_id)
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=workspace_path,
                status="eval_failed",
                checkpoint=checkpoint,
                train_metrics=train_metrics,
                cost={"seconds": time.time() - t0},
            )

        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=workspace_path,
            status="success",
            checkpoint=checkpoint,
            eval_metrics=eval_metrics,
            error_buckets=error_buckets,
            validity=ValidityReport(is_valid=True),
            train_metrics=train_metrics,
            cost={"seconds": time.time() - t0},
            artifacts={
                "script_path": str(script_path),
                "trial_dir": str(trial_dir),
            },
        )

    def _run_mock_trial(
        self,
        workspace: Any,
        node: TrainingSearchNode,
        config: OLMoCoreTrainingConfig,
        trial_dir: Path,
        t0: float,
    ) -> TrainingTrialResult:
        """Run a mock trial for testing without GPU."""
        import random

        ckpt_dir = Path(config.save_folder)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model.pt").write_text("mock_checkpoint")

        mock_loss = 2.5 + random.uniform(-0.5, 0.5)
        train_metrics = {
            "final_loss": mock_loss,
            "steps": config.max_steps,
            "lr": config.lr,
            "mock": True,
        }

        checkpoint = CheckpointRef(
            name=f"ckpt_{node.node_id}",
            path=str(ckpt_dir),
            kind="full_weights",
            metadata={"steps": config.max_steps},
        )

        eval_metrics = EvalMetrics(
            primary_metric_name="loss",
            primary_metric_value=mock_loss,
            maximize=False,
            secondary={"perplexity": 2 ** mock_loss},
        )

        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=str(workspace.root),
            status="success",
            checkpoint=checkpoint,
            eval_metrics=eval_metrics,
            error_buckets=ErrorBuckets(),
            validity=ValidityReport(is_valid=True),
            train_metrics=train_metrics,
            cost={"seconds": time.time() - t0},
            artifacts={"trial_dir": str(trial_dir), "mock": "true"},
        )

    def _launch_and_wait(
        self,
        script_path: Path,
        config: OLMoCoreTrainingConfig,
        budget: TrialBudget,
        trial_dir: Path,
    ) -> dict[str, Any]:
        """Launch the training script and wait for completion."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._olmo_core_path / "src")

        if self.use_slurm:
            return self._launch_slurm(script_path, config, budget, trial_dir, env)
        else:
            return self._launch_local(script_path, config, budget, trial_dir, env)

    def _launch_local(
        self,
        script_path: Path,
        config: OLMoCoreTrainingConfig,
        budget: TrialBudget,
        trial_dir: Path,
        env: dict[str, str],
    ) -> dict[str, Any]:
        """Launch via torchrun locally."""
        cmd = [
            "torchrun",
            f"--nproc-per-node={config.gpus_per_node}",
            str(script_path),
        ]

        timeout = budget.seconds if budget.seconds else 3600 * 24
        log_path = trial_dir / "train.log"

        logger.info("Launching local training: %s", " ".join(cmd))
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(trial_dir),
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=30)
                logger.warning("Training timed out after %.0f seconds", timeout)

        metrics = self._collect_metrics_from_log(log_path)
        metrics["return_code"] = proc.returncode
        metrics["log_path"] = str(log_path)
        return metrics

    def _launch_slurm(
        self,
        script_path: Path,
        config: OLMoCoreTrainingConfig,
        budget: TrialBudget,
        trial_dir: Path,
        env: dict[str, str],
    ) -> dict[str, Any]:
        """Launch via SLURM sbatch."""
        sbatch_script = self._build_sbatch_script(script_path, config, budget, trial_dir)
        sbatch_path = trial_dir / "submit.sbatch"
        sbatch_path.write_text(sbatch_script)

        result = subprocess.run(
            ["sbatch", str(sbatch_path)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")

        import re
        match = re.search(r"(\d+)", result.stdout)
        job_id = match.group(1) if match else "unknown"
        logger.info("Submitted SLURM job: %s", job_id)

        self._wait_for_slurm_job(job_id, budget)

        metrics = self._collect_metrics_from_log(trial_dir / f"slurm-{job_id}.out")
        metrics["slurm_job_id"] = job_id
        return metrics

    def _build_sbatch_script(
        self,
        script_path: Path,
        config: OLMoCoreTrainingConfig,
        budget: TrialBudget,
        trial_dir: Path,
    ) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=a-evolve-olmo",
            f"#SBATCH --nodes={config.num_nodes}",
            f"#SBATCH --gpus-per-node={config.gpus_per_node}",
            f"#SBATCH --ntasks-per-node={config.gpus_per_node}",
            f"#SBATCH --output={trial_dir}/slurm-%j.out",
            f"#SBATCH --error={trial_dir}/slurm-%j.err",
        ]
        if self.slurm_partition:
            lines.append(f"#SBATCH --partition={self.slurm_partition}")
        if self.slurm_account:
            lines.append(f"#SBATCH --account={self.slurm_account}")
        if budget.seconds:
            hours = int(budget.seconds / 3600) + 1
            lines.append(f"#SBATCH --time={hours}:00:00")

        lines.extend([
            "",
            f"export PYTHONPATH={self._olmo_core_path / 'src'}",
            "",
            f"torchrun --nproc-per-node={config.gpus_per_node} "
            f"--nnodes={config.num_nodes} "
            f"--rdzv-backend=c10d "
            f"--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT "
            f"{script_path}",
        ])
        return "\n".join(lines)

    def _wait_for_slurm_job(self, job_id: str, budget: TrialBudget) -> None:
        """Poll SLURM until job completes or budget expires."""
        timeout = budget.seconds if budget.seconds else 3600 * 72
        start = time.time()
        poll_interval = 30

        while time.time() - start < timeout:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            state = result.stdout.strip()
            if not state or state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                break
            time.sleep(poll_interval)

    def _find_latest_checkpoint(self, save_folder: str) -> CheckpointRef | None:
        """Find the latest checkpoint in the save folder."""
        ckpt_dir = Path(save_folder)
        if not ckpt_dir.exists():
            return None

        candidates = sorted(ckpt_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if candidate.is_dir() or candidate.suffix in (".pt", ".safetensors"):
                return CheckpointRef(
                    name=candidate.name,
                    path=str(candidate),
                    kind="full_weights",
                    metadata={"save_folder": save_folder},
                )
        return None

    def _run_evaluation(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        benchmark: Any,
    ) -> tuple[EvalMetrics, ErrorBuckets]:
        """Run evaluation using the benchmark adapter.

        For pre-training, evaluation is typically loss on a held-out set.
        The benchmark can implement custom eval logic.
        """
        if hasattr(benchmark, "evaluate"):
            result_dir = benchmark.evaluate(workspace, checkpoint, self, "dev")
            metrics = benchmark.parse_metrics(result_dir)
            errors = benchmark.analyze_errors(result_dir, metrics)
            return metrics, errors

        # Fallback: use training metrics as eval proxy
        train_metrics = self._collect_metrics_from_log(
            Path(checkpoint.path).parent.parent / "train.log"
        )
        loss = train_metrics.get("final_loss", train_metrics.get("loss", 3.0))
        return (
            EvalMetrics(
                primary_metric_name="loss",
                primary_metric_value=float(loss),
                maximize=False,
            ),
            ErrorBuckets(),
        )

    def _collect_metrics_from_log(self, log_path: Path) -> dict[str, Any]:
        """Parse training metrics from log output."""
        metrics: dict[str, Any] = {}
        if not log_path.exists():
            return metrics

        try:
            text = log_path.read_text()
            import re
            loss_matches = re.findall(r"loss[:\s=]+([\d.]+)", text, re.IGNORECASE)
            if loss_matches:
                metrics["final_loss"] = float(loss_matches[-1])

            step_matches = re.findall(r"step[:\s=]+(\d+)", text, re.IGNORECASE)
            if step_matches:
                metrics["final_step"] = int(step_matches[-1])

            throughput_matches = re.findall(
                r"throughput[:\s=]+([\d.]+)", text, re.IGNORECASE
            )
            if throughput_matches:
                metrics["throughput"] = float(throughput_matches[-1])
        except Exception as exc:
            metrics["parse_error"] = str(exc)

        return metrics
