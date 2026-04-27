"""Single-node TinkerLite backend.

PR2 delivers a mock-capable shell that honors the protocol shape. PR7 fills in
the real SFT pipeline; until then the backend relies on MockTrainingClient to
keep the API exercisable on CPU.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from ...training.runners.data_worker import render_datums
from ...training.runners.eval_worker import run_eval_plan as _run_eval_plan
from ...training.runners.pack_adapter_worker import pack_adapter
from ...training.runners.train_worker import run_sft_stage
from ...training.types import (
    CheckpointRef,
    EvalMetrics,
    ErrorBuckets,
    EvalPlan,
    TrainingSearchNode,
    TrainingTrialResult,
    TrialBudget,
    ValidityReport,
)
from .mock_clients import MockSamplingClient, MockTrainingClient

logger = logging.getLogger(__name__)


class SingleNodeTinkerLiteBackend:
    name = "h200_single_node"

    def __init__(self, mock: bool = True) -> None:
        # `mock=True` is PR2's default — PR7 flips this to `False` for real
        # training when a GPU is available.
        self.mock = mock
        # These are set during ``run_trial`` so that ``run_eval_plan`` (called
        # via ``benchmark.evaluate(workspace, checkpoint, backend, split)``)
        # can reach the current workspace + benchmark without threading them
        # through the EvalPlan dataclass.
        self._current_workspace: Any | None = None
        self._current_benchmark: Any | None = None
        self._current_split: str | None = None

    # ── Factory methods ──────────────────────────────────────────────

    def create_training_client(
        self,
        workspace: Any,
        checkpoint: CheckpointRef | None = None,  # noqa: ARG002 (future use)
    ) -> MockTrainingClient:
        return MockTrainingClient(Path(workspace.root))

    def create_sampling_client(
        self,
        workspace: Any,  # noqa: ARG002
        checkpoint: CheckpointRef,  # noqa: ARG002
    ) -> MockSamplingClient:
        return MockSamplingClient()

    def run_eval_plan(self, plan: EvalPlan) -> Path:
        return _run_eval_plan(
            plan,
            smoke=self.mock,
            benchmark=self._current_benchmark,
            workspace=self._current_workspace,
            split=self._current_split,
        )

    # ── Trial ───────────────────────────────────────────────────────

    def run_trial(
        self,
        workspace: Any,
        node: TrainingSearchNode,
        budget: TrialBudget,
        benchmark: Any,
    ) -> TrainingTrialResult:
        t0 = time.time()
        workspace_path = str(workspace.root)
        self._current_workspace = workspace
        self._current_benchmark = benchmark
        self._current_split = _load_default_split(workspace)
        try:
            pipeline = _load_pipeline(workspace)
        except FileNotFoundError as exc:
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=workspace_path,
                status="train_failed",
                train_metrics={"error": f"pipeline missing: {exc}"},
            )

        try:
            seed_ckpt = _seed_adapter_ref(workspace)
            if seed_ckpt is not None and not pipeline.get("override_seed_adapter"):
                # Eval-only cycle: use the pre-provisioned adapter directly.
                checkpoint = seed_ckpt
                train_metrics = {
                    "stage": "seed_adapter_passthrough",
                    "adapter": seed_ckpt.path,
                }
            else:
                checkpoint, train_metrics = self._run_pipeline(workspace, pipeline, budget)
        except Exception as exc:  # train crashed
            logger.exception("train_worker crashed for node %s", node.node_id)
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=workspace_path,
                status="train_failed",
                train_metrics={"error": repr(exc)},
                cost={"seconds": time.time() - t0},
            )

        try:
            eval_result = _run_evaluation(workspace, checkpoint, benchmark, self)
        except Exception as exc:  # eval crashed
            logger.exception("eval_worker crashed for node %s", node.node_id)
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=workspace_path,
                status="eval_failed",
                checkpoint=checkpoint,
                train_metrics=train_metrics,
                cost={"seconds": time.time() - t0},
                artifacts={"pipeline": json.dumps(pipeline)},
            )

        metrics, error_buckets = eval_result
        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=workspace_path,
            status="success",
            checkpoint=checkpoint,
            eval_metrics=metrics,
            error_buckets=error_buckets,
            validity=ValidityReport(is_valid=True),
            train_metrics=train_metrics,
            cost={"seconds": time.time() - t0},
            artifacts={
                "workspace_path": workspace_path,
                "pipeline": json.dumps(pipeline),
            },
        )

    # ── Pipeline execution (PR2: mock) ───────────────────────────────

    def _run_pipeline(
        self,
        workspace: Any,
        pipeline: dict,
        budget: TrialBudget,
    ) -> tuple[CheckpointRef, dict[str, Any]]:
        stages = pipeline.get("stages", [])
        last_ckpt: CheckpointRef | None = None
        aggregated: dict[str, Any] = {"pipeline_stages": len(stages), "stage_metrics": []}
        optimizer = _load_yaml_safely(Path(workspace.root) / "train" / "optimizer.yaml")
        start = time.time()

        for stage in stages:
            if not stage.get("enabled", True):
                continue
            stype = stage.get("type")
            if stype != "sft":
                # Non-SFT stages are no-ops for PR7; PR9+ owns RL paths.
                continue
            remaining = budget.seconds - (time.time() - start) if budget.seconds else None
            # Smoke path still uses the mock Datum iterable; real path pulls
            # its tokenized dataset from ``render_hf_dataset`` inside
            # ``train_worker`` and ignores the ``datums`` arg.
            datums = list(render_datums(workspace, smoke=self.mock)) if self.mock else None
            ckpt, stage_metrics = run_sft_stage(
                workspace,
                stage,
                datums,
                optimizer=optimizer,
                smoke=self.mock,
                budget_seconds=remaining,
            )
            aggregated["stage_metrics"].append(stage_metrics)
            last_ckpt = ckpt

        if last_ckpt is None:
            # No SFT stage ran — emit a shell checkpoint so downstream eval
            # can still validate the status code path.
            shell = MockTrainingClient(Path(workspace.root))
            last_ckpt = shell.save_weights_for_sampler(name="empty")

        # Real-path guard: the real SFT path writes ``adapter_config.json``
        # directly via ``trainer.save_model``; the mock path writes a fake
        # ``state.json`` / ``weights.json``. Only re-pack when the mock path
        # was used — repacking a real adapter would discard its config.
        if self.mock:
            adapter = pack_adapter(
                last_ckpt,
                target_root=Path(workspace.root) / "checkpoints" / "adapters",
                adapter_name=last_ckpt.name,
                metadata={"pipeline_stages": len(stages)},
            )
        else:
            # Validity guard: the real trainer must have produced an adapter
            # directory with ``adapter_config.json``. Missing either is a
            # hard failure — raise so ``run_trial`` catches and flips the
            # trial status to ``train_failed``.
            adapter_dir = Path(last_ckpt.path)
            if not adapter_dir.is_dir() or not (adapter_dir / "adapter_config.json").is_file():
                raise RuntimeError(
                    f"Expected adapter_config.json under {adapter_dir} after real SFT"
                )
            adapter = last_ckpt
        aggregated["total_seconds"] = time.time() - start
        return adapter, aggregated


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_pipeline(workspace: Any) -> dict:
    path = Path(workspace.root) / "train" / "pipeline.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_yaml_safely(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:  # pragma: no cover — best-effort config load
        return {}


def _load_default_split(workspace: Any) -> str:
    cfg = _load_yaml_safely(Path(workspace.root) / "eval" / "kaggle_eval.yaml")
    return str(cfg.get("default_split", "local_holdout_small"))


def _seed_adapter_ref(workspace: Any) -> CheckpointRef | None:
    """If ``model/adapter.yaml::seed_adapter_path`` is set, return a CheckpointRef.

    Eval-only cycles use this to skip training and evaluate an existing
    adapter (e.g. the ``E-28`` baseline).
    """
    adapter_cfg = _load_yaml_safely(Path(workspace.root) / "model" / "adapter.yaml")
    seed = adapter_cfg.get("seed_adapter_path")
    if not seed:
        return None
    path = Path(seed)
    if not path.is_absolute():
        path = (Path(workspace.root) / seed).resolve()
    if not path.is_dir():
        return None
    return CheckpointRef(
        name=adapter_cfg.get("seed_adapter_name", "seed_adapter"),
        path=str(path),
        kind="adapter",
        metadata={"source": "seed_adapter_path"},
    )


def _run_evaluation(
    workspace: Any,
    checkpoint: CheckpointRef,
    benchmark: Any,
    backend: Any,
) -> tuple[EvalMetrics, ErrorBuckets]:
    """Drive benchmark evaluation, then parse metrics + error buckets."""
    split = "local_holdout_small"
    out: Path | None = None

    if hasattr(benchmark, "build_eval_plan") and hasattr(benchmark, "evaluate"):
        try:
            result_dir = benchmark.evaluate(workspace, checkpoint, backend, split)
            out = Path(result_dir) if result_dir else None
        except NotImplementedError:
            out = None

    if out is None or not out.exists():
        # Benchmarks that don't implement build_eval_plan (e.g. test fakes).
        out = Path(workspace.root) / "evolution" / "eval" / checkpoint.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({"primary": 0.0, "n": 0}))
        (out / "predictions.jsonl").write_text("")

    metric_spec_name = "primary"
    metric_spec_max = True
    if hasattr(benchmark, "primary_metric"):
        spec = benchmark.primary_metric()
        metric_spec_name = spec.name
        metric_spec_max = spec.maximize

    metrics: EvalMetrics = EvalMetrics(
        primary_metric_name=metric_spec_name,
        primary_metric_value=0.0,
        maximize=metric_spec_max,
    )
    buckets = ErrorBuckets(counts={})
    if hasattr(benchmark, "parse_metrics"):
        try:
            metrics = benchmark.parse_metrics(out)
        except Exception:  # pragma: no cover — best-effort in mocks
            pass
    if hasattr(benchmark, "analyze_errors"):
        try:
            buckets = benchmark.analyze_errors(out, metrics)
        except Exception:  # pragma: no cover
            pass
    return metrics, buckets


__all__ = ["SingleNodeTinkerLiteBackend"]
