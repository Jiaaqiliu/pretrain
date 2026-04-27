"""NemoReasonerBenchmark — training benchmark for a reasoning eval set.

Owns: primary metric, eval protocol, metric parsing, error taxonomy, validity.
Does **not** compute reward or pick incumbents.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from ..training.types import (
    CheckpointRef,
    ErrorBuckets,
    EvalMetrics,
    EvalPlan,
    MetricSpec,
    TrainingTrialResult,
    ValidityReport,
)


DEFAULT_PRIMARY_METRIC_NAME = "local_holdout_pass_at_1"


class NemoReasonerBenchmark:
    name = "nemo_reasoner"

    def __init__(self) -> None:
        pass

    # ── Metric spec ─────────────────────────────────────────────────

    def primary_metric(self) -> MetricSpec:
        return MetricSpec(
            name=DEFAULT_PRIMARY_METRIC_NAME,
            maximize=True,
            higher_is_better=True,
        )

    # ── Eval plan / execution ───────────────────────────────────────

    def build_eval_plan(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        split: str,
    ) -> EvalPlan:
        output_dir = (
            Path(workspace.root)
            / "evolution"
            / "eval"
            / checkpoint.name
            / split
        )
        return EvalPlan(
            benchmark_name=self.name,
            split=split,
            checkpoint=checkpoint,
            config_path=str(Path(workspace.root) / "eval" / "local_splits.yaml"),
            output_dir=str(output_dir),
            generation_config={"temperature": 0.0, "top_p": 1.0, "max_tokens": 8192, "n": 1},
        )

    def evaluate(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        backend: Any,
        split: str,
    ) -> Path:
        plan = self.build_eval_plan(workspace, checkpoint, split)
        return backend.run_eval_plan(plan)

    # ── Metric / error parsing ──────────────────────────────────────

    def parse_metrics(self, result_dir: Path) -> EvalMetrics:
        result_dir = Path(result_dir)
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        with open(metrics_path) as f:
            raw = json.load(f)
        # Be permissive: accept either the primary metric key or "primary".
        primary = raw.get(DEFAULT_PRIMARY_METRIC_NAME, raw.get("primary"))
        if primary is None:
            raise KeyError(
                f"Missing primary metric '{DEFAULT_PRIMARY_METRIC_NAME}' in {metrics_path}"
            )
        secondary = {
            k: float(v)
            for k, v in raw.items()
            if k not in {DEFAULT_PRIMARY_METRIC_NAME, "primary"} and isinstance(v, (int, float))
        }
        return EvalMetrics(
            primary_metric_name=DEFAULT_PRIMARY_METRIC_NAME,
            primary_metric_value=float(primary),
            maximize=True,
            secondary=secondary,
        )

    def analyze_errors(self, result_dir: Path, metrics: EvalMetrics) -> ErrorBuckets:
        result_dir = Path(result_dir)
        predictions_path = result_dir / "predictions.jsonl"
        if not predictions_path.exists():
            return ErrorBuckets(counts={})
        counts: dict[str, int] = {}
        examples: dict[str, list[dict]] = {}
        with open(predictions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counts["eval_runtime_error"] = counts.get("eval_runtime_error", 0) + 1
                    continue
                bucket = _classify(row)
                if bucket is None:
                    continue
                counts[bucket] = counts.get(bucket, 0) + 1
                examples.setdefault(bucket, [])
                if len(examples[bucket]) < 3:
                    examples[bucket].append(row)
        return ErrorBuckets(counts=counts, examples=examples)

    # ── Validity ────────────────────────────────────────────────────

    def check_validity(
        self,
        workspace: Any,
        trial_result: TrainingTrialResult,
    ) -> ValidityReport:
        flags: dict[str, Any] = {}

        # 1. checkpoint missing / adapter cannot be loaded
        ckpt = trial_result.checkpoint
        if ckpt is None or not ckpt.path:
            return ValidityReport(is_valid=False, hard_fail_reason="checkpoint_missing", flags=flags)
        if not Path(ckpt.path).exists():
            return ValidityReport(is_valid=False, hard_fail_reason="adapter_load_failed", flags=flags)

        # 2. training or eval crashed
        if trial_result.status in {"train_failed", "eval_failed", "invalid_adapter", "over_budget"}:
            return ValidityReport(
                is_valid=False,
                hard_fail_reason=trial_result.status,
                flags=flags,
            )

        # 3. missing metrics / NaN metric
        metrics = trial_result.eval_metrics
        if metrics is None:
            return ValidityReport(is_valid=False, hard_fail_reason="metrics_missing", flags=flags)
        if math.isnan(metrics.primary_metric_value):
            return ValidityReport(is_valid=False, hard_fail_reason="metric_nan", flags=flags)

        # 4. protected eval split modified (best-effort: compare mtime/size if available)
        protected = Path(workspace.root) / "eval" / "local_splits.yaml"
        if protected.exists():
            # Smoke check only; real tamper detection lives in the fingerprint.
            flags["protected_split_present"] = True

        # 5. adapter config violations (look for illegal rank)
        adapter_cfg_path = Path(workspace.root) / "model" / "adapter.yaml"
        if adapter_cfg_path.exists():
            try:
                with open(adapter_cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                rank = cfg.get("rank")
                if rank is not None and int(rank) <= 0:
                    return ValidityReport(
                        is_valid=False, hard_fail_reason="lora_rank_invalid", flags=flags
                    )
            except Exception as exc:
                return ValidityReport(
                    is_valid=False,
                    hard_fail_reason=f"adapter_config_error:{exc}",
                    flags=flags,
                )

        return ValidityReport(is_valid=True, flags=flags)


def _classify(row: dict) -> str | None:
    """Map a prediction row to an error bucket, or None if correct."""
    if row.get("is_correct") is True:
        return None
    # Prefer explicit label if provided.
    label = row.get("error_bucket")
    if label:
        return str(label)
    # Fallback heuristics — check explicit flags before falling back to
    # answer-extraction inference (a missing `answer` field is the weakest
    # signal, so it goes last).
    if row.get("format_error"):
        return "format_error"
    if row.get("overlong"):
        return "overlong_reasoning"
    if "answer" in row and row.get("answer") is None:
        return "answer_extraction_fail"
    return "wrong_rule"


__all__ = ["NemoReasonerBenchmark"]
