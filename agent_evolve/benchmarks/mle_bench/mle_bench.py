"""MLEBenchAdapter — training benchmark for MLE-Bench evaluation.

MLE-Bench (https://github.com/openai/mle-bench) evaluates agent performance
on real Kaggle competition tasks. This adapter:

* Defines primary metric (e.g., average competition score or success rate)
* Builds evaluation plans that run agent on Kaggle tasks
* Parses metrics from MLE-Bench output
* Analyzes error types (runtime errors, API failures, incorrect submissions)
* Validates training configurations

Two modes:
* **Smoke mode** (default): Small subset of tasks for quick validation
* **Full mode**: Complete MLE-Bench evaluation suite

Does **not** compute reward or pick incumbents (that's MCGS's job).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ...model.types import (
    CheckpointRef,
    ErrorBuckets,
    EvalMetrics,
    EvalPlan,
    MetricSpec,
    TrainingTrialResult,
    ValidityReport,
)


# Primary metric defaults
DEFAULT_PRIMARY_METRIC_NAME = "mle_bench_success_rate"
FULL_PRIMARY_METRIC_NAME = "mle_bench_avg_score"


@dataclass
class MLEBenchConfig:
    """Configuration loaded from workspace eval/mle_bench_eval.yaml."""

    split: str = "validation"
    limit: int | None = None
    timeout_per_task: int = 3600  # 1 hour per task
    max_concurrent: int = 1

    @classmethod
    def from_yaml(cls, path: Path) -> MLEBenchConfig:
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class MLEBenchAdapter:
    """Training benchmark adapter for MLE-Bench."""

    name = "mle_bench"

    def primary_metric(self) -> MetricSpec:
        """Return the primary metric specification."""
        # NOTE: an earlier mle_training revision passed ``valid_range=(0.0,
        # 1.0)``, but the canonical ``MetricSpec`` in
        # ``training/types.py`` only declares ``name / maximize /
        # higher_is_better``. The range is just metadata anyway — drop it
        # to keep the benchmark instantiable.
        return MetricSpec(
            name=DEFAULT_PRIMARY_METRIC_NAME,
            higher_is_better=True,
        )

    def build_eval_plan(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        split: str,
    ) -> EvalPlan:
        """Build evaluation plan for MLE-Bench tasks.

        Args:
            workspace: Training workspace containing eval config
            checkpoint: Trained model checkpoint/adapter reference
            split: Eval split name (e.g., "validation", "test")

        Returns:
            EvalPlan with evaluation configuration
        """
        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)
        eval_dir = workspace_path / "eval"
        model_dir = workspace_path / "model"

        # Load model config to get data paths
        model_config_path = model_dir / "config.yaml"
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)

        # For AutoML/sklearn backend, store data paths in metadata
        return EvalPlan(
            benchmark_name="mle_bench",
            split=split,
            checkpoint=checkpoint,
            config_path=str(model_config_path),
            output_dir=str(workspace_path / "evolution" / "eval" / checkpoint.kind / split),
            generation_config={},
            metadata={
                "test_data_path": str(workspace_path / "data" / model_config.get("test_data", "test.csv")),
                "id_column": model_config.get("id_column", "PassengerId"),
                "target_column": model_config.get("target_column", "target"),
                "competition_id": model_config.get("competition_id", "unknown"),
            },
        )

    def _build_task_prompt(self, task: dict) -> str:
        """Build prompt for a single MLE-Bench task."""
        prompt_parts = [
            "You are a machine learning engineer working on a Kaggle competition.",
            "",
            f"## Competition: {task.get('name', 'Unknown')}",
            "",
            f"### Description:",
            task.get("description", ""),
            "",
            "### Task:",
            "1. Analyze the provided data",
            "2. Build and train a model",
            "3. Generate predictions for the test set",
            "4. Format your submission according to the requirements",
            "",
            "Please provide your complete solution code and approach.",
            "<solution>",
        ]
        return "\n".join(prompt_parts)

    def evaluate(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        backend: Any,
        split: str,
    ) -> Path:
        """Run evaluation and return result directory.

        Args:
            workspace: Training workspace
            checkpoint: Model checkpoint/adapter
            backend: Execution backend
            split: Eval split name

        Returns:
            Path to directory containing metrics.json and predictions
        """
        eval_plan = self.build_eval_plan(workspace, checkpoint, split)
        result_dir = backend.run_eval_plan(
            workspace=workspace,
            checkpoint=checkpoint,
            plan=eval_plan,
        )

        # For sklearn backend, we need to grade the submission
        if hasattr(backend, '__class__') and 'Sklearn' in backend.__class__.__name__:
            self._grade_submission(workspace, result_dir, split)

        return result_dir

    def _grade_submission(self, workspace: Any, result_dir: Path, split: str):
        """Grade submission using MLE-Bench grader.

        This is called for sklearn backend to compute actual Kaggle scores.
        """
        submission_path = result_dir / "submission.csv"
        if not submission_path.exists():
            return

        # Get competition ID from workspace
        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)
        config_path = workspace_path / "model" / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        competition_id = config.get("competition_id", "unknown")

        try:
            # Use mlebench grader directly (more reliable than CLI)
            from mlebench.registry import registry
            from mlebench.utils import load_answers, read_csv

            competition = registry.get_competition(competition_id)
            submission_df = read_csv(submission_path)
            answers = load_answers(competition.answers)

            # Grade submission
            score = competition.grader(submission_df, answers)

            # Save metrics
            metrics = {
                "competition_id": competition_id,
                "mle_bench_score": float(score),
                "primary_metric": "mle_bench_score",
                "mle_bench_success_rate": 1.0 if score > 0 else 0.0,  # Success if valid score
            }

            # If backend wrote a cv_mean.json marker, attach CV metrics as SECONDARY
            # signal only. We do NOT change primary_metric:
            #
            # CV-mean is computed on full train via K-fold OOF; Kaggle score is on
            # a ~870-row public holdout. The two scales differ (CV is often ~0.02-0.03
            # lower than the holdout score for the same model). Replacing the primary
            # metric with CV-mean breaks MCGS comparisons across nodes with different
            # CV settings. Instead, expose CV as secondary so LLM/MCGS can use it as
            # a robustness / overfitting signal without scale confusion.
            cv_marker = result_dir / "cv_mean.json"
            if cv_marker.exists():
                with open(cv_marker) as cf:
                    cv_data = json.load(cf)
                cv_mean = cv_data.get("cv_mean_accuracy")
                if cv_mean is not None:
                    metrics["cv_mean_accuracy"] = float(cv_mean)
                    metrics["cv_std"] = float(cv_data.get("cv_std", 0.0))
                    metrics["cv_n_splits"] = cv_data.get("n_splits")
                    print(f"  CV-mean={cv_mean:.5f} ± {metrics['cv_std']:.5f} "
                          f"(secondary; primary stays mle_bench_score={score:.5f})")

            # Ensemble marker — record strategy + member count as secondary info
            ens_marker = result_dir / "ensemble.json"
            if ens_marker.exists():
                with open(ens_marker) as ef:
                    ens_data = json.load(ef)
                metrics["ensemble_strategy"] = ens_data.get("strategy")
                metrics["ensemble_n_members"] = ens_data.get("n_members")
                metrics["ensemble_member_types"] = ens_data.get("member_types")

            metrics_path = result_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"✓ Graded {competition_id}: score = {score:.5f}")

        except Exception as e:
            print(f"Warning: Failed to grade submission: {e}")
            # Create fallback metrics
            metrics_path = result_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({
                    "competition_id": competition_id,
                    "mle_bench_score": 0.0,
                    "error": str(e),
                }, f)

    def parse_metrics(self, result_dir: Path) -> EvalMetrics:
        """Parse evaluation metrics from result directory.

        Expects result_dir/metrics.json with:
        {
            "competition_id": "spaceship-titanic",
            "mle_bench_score": 0.748,
            "mle_bench_success_rate": 1.0
        }
        """
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            return EvalMetrics(
                primary_metric_name=DEFAULT_PRIMARY_METRIC_NAME,
                primary_metric_value=0.0,
                secondary={},
            )

        with open(metrics_path) as f:
            data = json.load(f)

        # Use mle_bench_score as primary metric
        primary_name = data.get("primary_metric", "mle_bench_score")
        primary_value = data.get(primary_name, 0.0)

        return EvalMetrics(
            primary_metric_name=primary_name,
            primary_metric_value=primary_value,
            maximize=True,
            secondary={
                k: v for k, v in data.items()
                if k not in ["primary_metric", primary_name]
            },
        )

    def analyze_errors(
        self,
        result_dir: Path,
        metrics: EvalMetrics,
    ) -> ErrorBuckets:
        """Analyze error patterns from evaluation results.

        Error categories:
        - runtime_error: Code execution failures
        - api_error: LLM API failures (rate limits, timeouts)
        - format_error: Invalid submission format
        - timeout: Task exceeded time limit
        - low_score: Completed but scored below threshold
        """
        predictions_path = result_dir / "predictions.jsonl"
        if not predictions_path.exists():
            return ErrorBuckets(
                counts={},
                examples={},
            )

        error_counts: dict[str, int] = {
            "runtime_error": 0,
            "api_error": 0,
            "format_error": 0,
            "timeout": 0,
            "low_score": 0,
            "success": 0,
        }
        error_examples = []
        total = 0

        with open(predictions_path) as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                pred = json.loads(line)

                error_type = pred.get("error_type", "")
                if error_type:
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    if len(error_examples) < 10:  # Keep up to 10 examples
                        error_examples.append({
                            "task_id": pred.get("task_id"),
                            "error_type": error_type,
                            "error_message": pred.get("error_message", ""),
                        })
                elif pred.get("score", 0.0) < 0.3:
                    error_counts["low_score"] += 1
                else:
                    error_counts["success"] += 1

        return ErrorBuckets(
            counts=error_counts,
            examples={"all_errors": error_examples},
        )

    def check_validity(
        self,
        workspace: Any,
        trial_result: TrainingTrialResult,
    ) -> ValidityReport:
        """Check if training result satisfies constraints.

        MLE-Bench constraints:
        - Model must fit within memory limits
        - Inference time per task must be reasonable
        - Adapter rank constraints (if using LoRA)
        """
        # Check if evaluation completed
        if trial_result.status != "success":
            return ValidityReport(
                is_valid=False,
                hard_fail_reason=f"Trial did not complete: {trial_result.status}",
                flags={},
            )

        # Check adapter rank if using LoRA (not applicable for sklearn models)
        flags = {}

        # Check success rate threshold
        if trial_result.eval_metrics:
            success_rate = trial_result.eval_metrics.primary_metric_value
            if success_rate < 0.1:
                flags["low_success_rate"] = success_rate

        return ValidityReport(
            is_valid=True,
            hard_fail_reason=None,
            flags=flags,
        )


__all__ = ["MLEBenchAdapter", "MLEBenchConfig"]
