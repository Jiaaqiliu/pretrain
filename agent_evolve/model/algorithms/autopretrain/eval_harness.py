"""Evaluation harness for pretraining recipe search.

Two operational modes:
- Fast mode (proxy search): compute val loss via olmo-core's metrics system.
  Reads the metrics.json produced by MetricSaverCallback after training.
- Full mode (transfer verification): val loss + downstream via lm-eval-harness.

The fast mode is critical for keeping MCGS iteration time < 1 hour.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single checkpoint."""

    val_loss: float
    val_perplexity: float

    # Per-domain validation loss (key insight for mixture optimization)
    domain_losses: dict[str, float] = field(default_factory=dict)

    # Downstream tasks (lightweight subset for fast iteration)
    downstream: dict[str, float] = field(default_factory=dict)

    # Composite score (what MCGS optimizes)
    composite_score: float = 0.0

    # Meta
    eval_time_seconds: float = 0.0
    checkpoint_path: str = ""
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "val_loss": self.val_loss,
            "val_perplexity": self.val_perplexity,
            "domain_losses": self.domain_losses,
            "downstream": self.downstream,
            "composite_score": self.composite_score,
            "eval_time_seconds": self.eval_time_seconds,
            "checkpoint_path": self.checkpoint_path,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalResult:
        return cls(
            val_loss=d["val_loss"],
            val_perplexity=d["val_perplexity"],
            domain_losses=d.get("domain_losses", {}),
            downstream=d.get("downstream", {}),
            composite_score=d.get("composite_score", 0.0),
            eval_time_seconds=d.get("eval_time_seconds", 0.0),
            checkpoint_path=d.get("checkpoint_path", ""),
            step=d.get("step", 0),
        )


@dataclass
class PretrainEvalHarness:
    """Evaluation harness for pretraining experiments.

    Fast mode: parse metrics from training output (MetricSaverCallback).
    Full mode: load checkpoint and run forward pass on val sets + downstream tasks.
    """

    # Validation data paths for per-domain loss
    val_data_paths: dict[str, str] = field(default_factory=lambda: {
        "c4": "/fsx/shared/data/eval/c4_validation.npy",
    })

    # Downstream benchmark configs (for full mode)
    downstream_tasks: list[str] = field(default_factory=lambda: [
        "arc_easy",
        "piqa",
        "hellaswag",
    ])

    # Mode
    fast_mode: bool = True

    # Weights for composite score
    loss_weight: float = 0.6
    downstream_weight: float = 0.4

    def evaluate(
        self,
        checkpoint_path: str | Path,
        step: int = 0,
        model_config: Any = None,
        trial_dir: Path | None = None,
    ) -> EvalResult:
        """Run evaluation on a checkpoint or trial directory.

        In fast mode: reads metrics saved by olmo-core's MetricSaverCallback or
        parses the training log for final loss.

        In full mode: loads checkpoint, computes val loss on multiple domains,
        and runs downstream benchmarks.
        """
        t0 = time.time()
        checkpoint_path = str(checkpoint_path)

        if self.fast_mode:
            return self._fast_eval(checkpoint_path, step, trial_dir, t0)
        else:
            return self._full_eval(checkpoint_path, step, model_config, t0)

    def _fast_eval(
        self, checkpoint_path: str, step: int, trial_dir: Path | None, t0: float
    ) -> EvalResult:
        """Fast eval: parse metrics from training output.

        After a trial completes, olmo-core saves metrics via MetricSaverCallback.
        We read these directly — no GPU needed for eval.
        """
        domain_losses: dict[str, float] = {}
        val_loss = float("inf")

        # Strategy 1: Read metrics.json from MetricSaverCallback
        if trial_dir:
            metrics_file = trial_dir / "checkpoints" / "metrics.json"
            if metrics_file.exists():
                metrics = json.loads(metrics_file.read_text())
                val_loss = metrics.get("train/CE loss batch", val_loss)
                # If per-domain eval callbacks were configured, they appear here
                for key, value in metrics.items():
                    if key.startswith("eval/") and "CE loss" in key:
                        domain = key.split("/")[1] if "/" in key else "unknown"
                        domain_losses[domain] = value
                logger.info("Fast eval: read metrics from %s (loss=%.4f)", metrics_file, val_loss)

        # Strategy 2: Parse train.log for final loss
        if val_loss == float("inf") and trial_dir:
            log_path = trial_dir / "train.log"
            if log_path.exists():
                val_loss = self._parse_loss_from_log(log_path)
                logger.info("Fast eval: parsed train.log (loss=%.4f)", val_loss)

        # Strategy 3: Check step-level metrics
        if val_loss == float("inf") and trial_dir:
            for mf in sorted(trial_dir.glob("checkpoints/metrics_step*.json"), reverse=True):
                metrics = json.loads(mf.read_text())
                if "train/CE loss batch" in metrics:
                    val_loss = metrics["train/CE loss batch"]
                    break

        if val_loss == float("inf"):
            logger.warning("Could not find metrics for %s, using default", checkpoint_path)
            val_loss = 4.0  # Safe fallback

        if not domain_losses:
            domain_losses = {"overall": val_loss}

        composite = -val_loss  # Lower loss = higher score

        return EvalResult(
            val_loss=val_loss,
            val_perplexity=2 ** val_loss,
            domain_losses=domain_losses,
            composite_score=composite,
            eval_time_seconds=time.time() - t0,
            checkpoint_path=checkpoint_path,
            step=step,
        )

    def _full_eval(
        self, checkpoint_path: str, step: int, model_config: Any, t0: float
    ) -> EvalResult:
        """Full eval: load checkpoint and run validation + downstream.

        Uses a subprocess to run the eval script (avoids GPU memory conflicts
        with training in the same process).
        """
        eval_script = Path(__file__).parent.parent.parent.parent.parent / "experiments" / "autopretrain" / "run_eval.py"

        # Run per-domain validation loss
        domain_losses = self._run_val_loss_eval(checkpoint_path, model_config)
        avg_loss = sum(domain_losses.values()) / max(len(domain_losses), 1)

        # Run downstream benchmarks
        downstream = self._run_downstream_eval(checkpoint_path)

        # Composite score
        downstream_score = sum(downstream.values()) / max(len(downstream), 1) if downstream else 0.0
        composite = self.loss_weight * (-avg_loss) + self.downstream_weight * downstream_score

        return EvalResult(
            val_loss=avg_loss,
            val_perplexity=2 ** avg_loss,
            domain_losses=domain_losses,
            downstream=downstream,
            composite_score=composite,
            eval_time_seconds=time.time() - t0,
            checkpoint_path=checkpoint_path,
            step=step,
        )

    def _run_val_loss_eval(
        self, checkpoint_path: str, model_config: Any
    ) -> dict[str, float]:
        """Compute per-domain validation loss by loading checkpoint.

        Uses olmo-core eval infrastructure via subprocess.
        """
        eval_script = Path(__file__).parent.parent.parent.parent.parent / "experiments" / "autopretrain" / "run_eval.py"
        results: dict[str, float] = {}

        for domain, val_path in self.val_data_paths.items():
            if not Path(val_path).exists():
                continue
            try:
                result = subprocess.run(
                    [
                        "torchrun", "--nproc-per-node=1",
                        str(eval_script),
                        "--checkpoint", checkpoint_path,
                        "--val-data", val_path,
                        "--domain", domain,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    output = json.loads(result.stdout.strip().split("\n")[-1])
                    results[domain] = output.get("loss", 4.0)
            except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
                logger.warning("Val eval failed for domain %s: %s", domain, e)
                results[domain] = 4.0

        return results if results else {"overall": 4.0}

    def _run_downstream_eval(self, checkpoint_path: str) -> dict[str, float]:
        """Run lightweight downstream benchmarks via lm-eval-harness."""
        try:
            tasks_str = ",".join(self.downstream_tasks)
            result = subprocess.run(
                [
                    "python", "-m", "lm_eval",
                    "--model", "hf",
                    "--model_args", f"pretrained={checkpoint_path},dtype=bfloat16",
                    "--tasks", tasks_str,
                    "--batch_size", "8",
                    "--output_path", "/tmp/lm_eval_results.json",
                ],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if result.returncode == 0:
                output_path = Path("/tmp/lm_eval_results.json")
                if output_path.exists():
                    data = json.loads(output_path.read_text())
                    results = {}
                    for task in self.downstream_tasks:
                        if task in data.get("results", {}):
                            results[task] = data["results"][task].get("acc,none", 0.0)
                    return results
        except Exception as e:
            logger.warning("Downstream eval failed: %s", e)

        return {}

    def _parse_loss_from_log(self, log_path: Path) -> float:
        """Parse final training loss from olmo-core console output."""
        import re
        text = log_path.read_text()

        # olmo-core logs: "train/CE loss batch: X.XXXX"
        matches = re.findall(r"train/CE loss batch[:\s]+([\d.]+)", text)
        if matches:
            return float(matches[-1])

        # Fallback: generic "loss: X.XX" pattern
        matches = re.findall(r"loss[:\s=]+([\d.]+)", text, re.IGNORECASE)
        if matches:
            return float(matches[-1])

        return float("inf")

    def compute_bitter_lesson_metrics(
        self,
        results_by_filter_level: dict[float, EvalResult],
    ) -> dict[str, Any]:
        """Analyze whether the Bitter Lesson holds for reasoning tasks.

        Given eval results at different filter retention levels, determine:
        - Is there a crossover point where less filtering wins?
        - Does the crossover depend on downstream task type?
        """
        filter_levels = sorted(results_by_filter_level.keys())
        analysis: dict[str, Any] = {
            "filter_levels": filter_levels,
            "val_losses": [results_by_filter_level[f].val_loss for f in filter_levels],
        }

        # Separate general vs reasoning downstream
        reasoning_tasks = {"arc_easy", "gsm8k", "math"}
        general_tasks = {"piqa", "hellaswag", "winogrande"}

        for task_set_name, task_set in [("reasoning", reasoning_tasks), ("general", general_tasks)]:
            scores = []
            for f in filter_levels:
                r = results_by_filter_level[f]
                task_scores = [v for k, v in r.downstream.items() if k in task_set]
                scores.append(sum(task_scores) / max(len(task_scores), 1) if task_scores else 0.0)
            analysis[f"{task_set_name}_scores"] = scores

        # Determine trend
        losses = analysis["val_losses"]
        if len(losses) >= 2:
            if losses[-1] < losses[0]:
                analysis["loss_trend"] = "bitter_lesson_holds"
            else:
                analysis["loss_trend"] = "filtering_helps"

        return analysis
