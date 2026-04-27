"""NemoReasonerBenchmark — training benchmark for Nemotron reasoning eval.

Two modes:

* **Generic / smoke mode** (default): small ``local_holdout_small.jsonl`` on
  disk, primary metric ``local_holdout_pass_at_1``. Used by the PR8 seed and
  unit tests.
* **Kaggle mode**: activated when ``eval/kaggle_eval.yaml`` exists in the
  workspace. Loads the Nemotron-Reasoning dev split (``id,prompt,answer,domain``
  CSV) and scores with the verbatim Kaggle host metric (boxed-answer EM, with
  a relative-tolerance numeric fallback). Primary metric:
  ``kaggle_nemo_boxed_em``.

Owns: primary metric, eval protocol, metric parsing, error taxonomy, validity.
Does **not** compute reward or pick incumbents.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
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


# ── Primary-metric defaults ──────────────────────────────────────────────

DEFAULT_PRIMARY_METRIC_NAME = "local_holdout_pass_at_1"
KAGGLE_PRIMARY_METRIC_NAME = "kaggle_nemo_boxed_em"

# ── Verbatim host-metric helpers (mirrored from
#    ../nemotron-auto-research/scripts/eval_metric.py) ────────────────────

EVAL_INSTRUCTION_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_final_answer(text: str | None) -> str:
    """Byte-for-byte equivalent to eval_metric.extract_final_answer."""
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1].strip()

    numeric = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numeric:
        return numeric[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Byte-for-byte equivalent to eval_metric.verify."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def build_eval_prompt(raw_prompt: str, tokenizer: Any) -> str:
    user_content = raw_prompt + EVAL_INSTRUCTION_SUFFIX
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception:
        return user_content


# ── Dev-row datatype ─────────────────────────────────────────────────────

@dataclass
class DevRow:
    id: str
    prompt: str
    answer: str
    domain: str | None = None


def _load_dev_rows(path: Path) -> list[DevRow]:
    rows: list[DevRow] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                DevRow(
                    id=str(raw.get("id", "")),
                    prompt=str(raw.get("prompt", "")),
                    answer=str(raw.get("answer", "")),
                    domain=raw.get("domain") or None,
                )
            )
    return rows


# ── Benchmark class ──────────────────────────────────────────────────────

class NemoReasonerBenchmark:
    name = "nemo_reasoner"

    def __init__(self) -> None:
        self._cached_dev: tuple[Path, list[DevRow]] | None = None

    # ── Spec ────────────────────────────────────────────────────────

    def primary_metric(self, workspace: Any | None = None) -> MetricSpec:
        name = DEFAULT_PRIMARY_METRIC_NAME
        if workspace is not None:
            cfg = _load_kaggle_eval_config(workspace)
            if cfg:
                name = str(cfg.get("primary_metric_name", KAGGLE_PRIMARY_METRIC_NAME))
        return MetricSpec(name=name, maximize=True, higher_is_better=True)

    # ── Eval plan / execution ───────────────────────────────────────

    def build_eval_plan(
        self,
        workspace: Any,
        checkpoint: CheckpointRef,
        split: str,
    ) -> EvalPlan:
        cfg = _load_kaggle_eval_config(workspace)
        output_dir = (
            Path(workspace.root)
            / "evolution"
            / "eval"
            / checkpoint.name
            / split
        )
        # Resolve dev path: prefer splits map, fallback to legacy local_holdout.
        splits_cfg_path = Path(workspace.root) / "eval" / "local_splits.yaml"
        dev_rel: str | None = None
        if splits_cfg_path.exists():
            with open(splits_cfg_path) as f:
                raw = yaml.safe_load(f) or {}
            dev_rel = (raw.get("splits") or {}).get(split)
        dev_path = _resolve_split_path(workspace, dev_rel, split)

        gen_cfg = {
            "temperature": cfg.get("temperature", 0.0),
            "top_p": cfg.get("top_p", 1.0),
            "max_tokens": cfg.get("max_tokens", 3584),
            "max_model_len": cfg.get("max_model_len", 4096),
            "max_lora_rank": cfg.get("max_lora_rank", 32),
            "tensor_parallel_size": cfg.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.85),
            "max_num_seqs": cfg.get("max_num_seqs", 128),
            "seed": cfg.get("seed", 0),
            "model_path": cfg.get("model_path") or _model_path_from_base(workspace),
            "limit": cfg.get("limit"),
            "dev_path": str(dev_path),
            "primary_metric_name": cfg.get("primary_metric_name", KAGGLE_PRIMARY_METRIC_NAME)
            if cfg
            else DEFAULT_PRIMARY_METRIC_NAME,
        }
        return EvalPlan(
            benchmark_name=self.name,
            split=split,
            checkpoint=checkpoint,
            config_path=str(dev_path),
            output_dir=str(output_dir),
            generation_config=gen_cfg,
            metadata={"split": split, "kaggle_mode": bool(cfg)},
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

    # ── Dev-row access (used by the non-smoke eval worker) ──────────

    def load_dev_rows(self, workspace: Any, split: str) -> list[DevRow]:
        splits_cfg = Path(workspace.root) / "eval" / "local_splits.yaml"
        rel: str | None = None
        if splits_cfg.exists():
            with open(splits_cfg) as f:
                raw = yaml.safe_load(f) or {}
            rel = (raw.get("splits") or {}).get(split)
        dev_path = _resolve_split_path(workspace, rel, split)
        if self._cached_dev and self._cached_dev[0] == dev_path:
            return self._cached_dev[1]
        rows = _load_dev_rows(dev_path)
        self._cached_dev = (dev_path, rows)
        return rows

    def score_predictions(
        self,
        dev_rows: list[DevRow],
        raw_outputs: list[str],
    ) -> tuple[EvalMetrics, ErrorBuckets, list[dict]]:
        assert len(dev_rows) == len(raw_outputs)
        per_domain: dict[str, tuple[int, int]] = {}
        records: list[dict] = []
        total_correct = 0
        for row, text in zip(dev_rows, raw_outputs):
            pred = extract_final_answer(text)
            correct = bool(verify(row.answer, pred))
            total_correct += int(correct)
            domain = row.domain or "unknown"
            c, n = per_domain.get(domain, (0, 0))
            per_domain[domain] = (c + int(correct), n + 1)
            records.append(
                {
                    "id": row.id,
                    "domain": row.domain,
                    "answer": row.answer,
                    "prediction": pred,
                    "is_correct": correct,
                    "raw_output_len": len(text or ""),
                }
            )
        overall = total_correct / max(1, len(dev_rows))
        metrics = EvalMetrics(
            primary_metric_name=KAGGLE_PRIMARY_METRIC_NAME,
            primary_metric_value=overall,
            maximize=True,
            secondary={
                **{
                    f"domain.{d}.accuracy": (c / max(1, n))
                    for d, (c, n) in per_domain.items()
                },
                **{f"domain.{d}.n": float(n) for d, (_, n) in per_domain.items()},
                "n_eval": float(len(dev_rows)),
            },
        )
        buckets = self._analyze_records(records)
        return metrics, buckets, records

    # ── Parsing (smoke + Kaggle both land here) ─────────────────────

    def parse_metrics(self, result_dir: Path) -> EvalMetrics:
        result_dir = Path(result_dir)
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        with open(metrics_path) as f:
            raw = json.load(f)
        primary = (
            raw.get(KAGGLE_PRIMARY_METRIC_NAME)
            or raw.get(DEFAULT_PRIMARY_METRIC_NAME)
            or raw.get("overall_accuracy")
            or raw.get("primary")
        )
        if primary is None:
            raise KeyError(
                f"Missing primary metric in {metrics_path}: want one of "
                f"{KAGGLE_PRIMARY_METRIC_NAME!r}, {DEFAULT_PRIMARY_METRIC_NAME!r}, "
                "'overall_accuracy', 'primary'"
            )
        # Preserve whichever name the file declared, falling back to generic.
        primary_name = (
            KAGGLE_PRIMARY_METRIC_NAME
            if KAGGLE_PRIMARY_METRIC_NAME in raw or "overall_accuracy" in raw
            else DEFAULT_PRIMARY_METRIC_NAME
        )
        secondary: dict[str, float] = {}
        for k, v in raw.items():
            if k in {KAGGLE_PRIMARY_METRIC_NAME, DEFAULT_PRIMARY_METRIC_NAME, "primary",
                     "overall_accuracy", "per_domain", "adapter", "model_path", "seed"}:
                continue
            if isinstance(v, (int, float)):
                secondary[k] = float(v)
        for domain, payload in (raw.get("per_domain") or {}).items():
            if isinstance(payload, dict):
                if "accuracy" in payload:
                    secondary[f"domain.{domain}.accuracy"] = float(payload["accuracy"])
                if "n" in payload:
                    secondary[f"domain.{domain}.n"] = float(payload["n"])
        secondary.setdefault("n_eval", float(raw.get("n_eval", 0) or 0))
        return EvalMetrics(
            primary_metric_name=primary_name,
            primary_metric_value=float(primary),
            maximize=True,
            secondary=secondary,
        )

    def analyze_errors(self, result_dir: Path, metrics: EvalMetrics) -> ErrorBuckets:
        result_dir = Path(result_dir)
        preds_path = result_dir / "predictions.jsonl"
        if not preds_path.exists():
            return ErrorBuckets(counts={})
        counts: dict[str, int] = {}
        examples: dict[str, list[dict]] = {}
        with open(preds_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counts["eval_runtime_error"] = counts.get("eval_runtime_error", 0) + 1
                    continue
                if row.get("is_correct") is True:
                    continue
                bucket = self._classify(row)
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

        ckpt = trial_result.checkpoint
        if ckpt is None or not ckpt.path:
            return ValidityReport(
                is_valid=False, hard_fail_reason="checkpoint_missing", flags=flags
            )
        if not Path(ckpt.path).exists():
            return ValidityReport(
                is_valid=False, hard_fail_reason="adapter_load_failed", flags=flags
            )

        if trial_result.status in {"train_failed", "eval_failed", "invalid_adapter", "over_budget"}:
            return ValidityReport(
                is_valid=False, hard_fail_reason=trial_result.status, flags=flags
            )

        # Kaggle host rule: LoRA rank ≤ 32.
        adapter_cfg_json = Path(ckpt.path) / "adapter_config.json"
        if adapter_cfg_json.exists():
            try:
                with open(adapter_cfg_json) as f:
                    cfg = json.load(f)
                rank = int(cfg.get("r", cfg.get("rank", 0) or 0))
                flags["lora_rank"] = rank
                if rank > 32:
                    return ValidityReport(
                        is_valid=False,
                        hard_fail_reason=f"lora_rank_invalid:{rank}>32",
                        flags=flags,
                    )
            except Exception as exc:
                return ValidityReport(
                    is_valid=False,
                    hard_fail_reason=f"adapter_config_error:{exc}",
                    flags=flags,
                )

        metrics = trial_result.eval_metrics
        if metrics is None:
            return ValidityReport(
                is_valid=False, hard_fail_reason="metrics_missing", flags=flags
            )
        if math.isnan(metrics.primary_metric_value):
            return ValidityReport(
                is_valid=False, hard_fail_reason="metric_nan", flags=flags
            )

        protected = Path(workspace.root) / "eval" / "local_splits.yaml"
        if protected.exists():
            flags["protected_split_present"] = True

        adapter_yaml = Path(workspace.root) / "model" / "adapter.yaml"
        if adapter_yaml.exists():
            try:
                with open(adapter_yaml) as f:
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

    # ── Internals ────────────────────────────────────────────────────

    def _analyze_records(self, records: list[dict]) -> ErrorBuckets:
        counts: dict[str, int] = {}
        examples: dict[str, list[dict]] = {}
        for row in records:
            if row.get("is_correct") is True:
                continue
            bucket = self._classify(row)
            if bucket is None:
                continue
            counts[bucket] = counts.get(bucket, 0) + 1
            examples.setdefault(bucket, [])
            if len(examples[bucket]) < 3:
                examples[bucket].append(row)
        return ErrorBuckets(counts=counts, examples=examples)

    def _classify(self, row: dict) -> str | None:
        if row.get("is_correct") is True:
            return None
        label = row.get("error_bucket")
        if label:
            return str(label)
        if row.get("format_error"):
            return "format_error"
        if row.get("overlong") or row.get("raw_output_len", 0) > 20000:
            return "overlong_reasoning"
        pred = row.get("prediction")
        if pred is None or (isinstance(pred, str) and pred.strip() in {"", "NOT_FOUND"}):
            return "answer_extraction_fail"
        if "answer" in row and row.get("answer") is None:
            return "answer_extraction_fail"
        return "wrong_rule"


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_kaggle_eval_config(workspace: Any) -> dict:
    """Return the ``eval/kaggle_eval.yaml`` dict, or {} if it's absent."""
    path = Path(workspace.root) / "eval" / "kaggle_eval.yaml"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:  # pragma: no cover — best-effort config load
        return {}


def _model_path_from_base(workspace: Any) -> str | None:
    base = Path(workspace.root) / "model" / "base.yaml"
    if not base.exists():
        return None
    try:
        with open(base) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("path")
    except Exception:  # pragma: no cover
        return None


def _resolve_split_path(workspace: Any, rel: str | None, split: str) -> Path:
    candidates: list[Path] = []
    if rel is not None:
        p = Path(rel)
        candidates.append(p if p.is_absolute() else (Path(workspace.root) / rel))
    # Legacy fallbacks used by the PR8 seed workspace.
    candidates.append(Path(workspace.root) / "eval" / f"{split}.jsonl")
    candidates.append(Path(workspace.root) / "eval" / "local_holdout_small.jsonl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No dev split found for {split!r} under {workspace.root} "
        f"(tried: {[str(c) for c in candidates]})"
    )


__all__ = [
    "NemoReasonerBenchmark",
    "DEFAULT_PRIMARY_METRIC_NAME",
    "KAGGLE_PRIMARY_METRIC_NAME",
    "DevRow",
    "extract_final_answer",
    "verify",
    "build_eval_prompt",
    "EVAL_INSTRUCTION_SUFFIX",
]
