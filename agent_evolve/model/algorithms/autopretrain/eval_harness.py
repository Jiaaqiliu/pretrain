"""Evaluation harness for pretraining recipe search.

Evaluates a trained checkpoint on:
1. Validation loss (per-domain decomposition)
2. Downstream reasoning benchmarks (lightweight subset)
3. Custom metrics for Bitter Lesson analysis
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single checkpoint."""

    # Core metrics
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


@dataclass
class PretrainEvalHarness:
    """Evaluation harness for pretraining experiments.

    Supports two modes:
    - Fast mode (proxy search): val loss only, < 2 min per eval
    - Full mode (transfer verification): val loss + downstream, ~15 min per eval
    """

    # Validation data paths (per-domain for loss decomposition)
    val_data_paths: dict[str, str] = field(default_factory=lambda: {
        "c4": "/fsx/shared/data/eval/c4_validation.npy",
        "web": "/fsx/dev/jiaqi/data/olmo-3b-pretrain/web",
        "code": "/fsx/dev/jiaqi/data/olmo-3b-pretrain/code",
        "math": "/fsx/dev/jiaqi/data/olmo-3b-pretrain/math",
    })

    # Downstream benchmark configs
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
    ) -> EvalResult:
        """Run evaluation on a checkpoint.

        In fast mode, only computes validation loss (sufficient for proxy search).
        In full mode, also runs downstream benchmarks.
        """
        t0 = time.time()
        checkpoint_path = str(checkpoint_path)

        # 1. Compute validation loss (per-domain)
        domain_losses = self._compute_val_losses(checkpoint_path, model_config)
        avg_loss = sum(domain_losses.values()) / max(len(domain_losses), 1)

        # 2. Downstream benchmarks (if not fast mode)
        downstream = {}
        if not self.fast_mode:
            downstream = self._run_downstream(checkpoint_path, model_config)

        # 3. Compute composite score
        # Lower loss is better → negate for "higher is better" composite
        loss_score = -avg_loss
        downstream_score = sum(downstream.values()) / max(len(downstream), 1) if downstream else 0.0

        if self.fast_mode:
            composite = loss_score
        else:
            composite = self.loss_weight * loss_score + self.downstream_weight * downstream_score

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

    def _compute_val_losses(
        self, checkpoint_path: str, model_config: Any
    ) -> dict[str, float]:
        """Compute per-domain validation loss.

        This is the core eval for proxy search — fast and informative.
        Domain-decomposed loss tells us which domains need more/less data.
        """
        # TODO: Implement using olmo-core eval utilities
        # For now, return placeholder that will be replaced with real eval
        logger.info("Computing validation losses for %s", checkpoint_path)

        # Real implementation will:
        # 1. Load checkpoint
        # 2. For each domain val set, compute cross-entropy loss
        # 3. Return {domain: loss}

        # Placeholder for testing the search loop
        return {"c4": 3.0, "web": 3.1, "code": 3.5, "math": 4.0}

    def _run_downstream(
        self, checkpoint_path: str, model_config: Any
    ) -> dict[str, float]:
        """Run lightweight downstream benchmarks.

        Uses lm-evaluation-harness or olmo-core's built-in eval.
        Only run in full mode (transfer verification).
        """
        # TODO: Implement using lm-eval-harness or olmo-core eval
        logger.info("Running downstream benchmarks for %s", checkpoint_path)

        # Placeholder
        return {"arc_easy": 0.5, "piqa": 0.65, "hellaswag": 0.4}

    def compute_bitter_lesson_metrics(
        self,
        results_by_filter_level: dict[float, EvalResult],
    ) -> dict[str, Any]:
        """Analyze whether the Bitter Lesson holds for reasoning tasks.

        Given eval results at different filter levels, determine:
        - Is there a crossover point where less filtering wins?
        - Does the crossover depend on downstream task type?
        """
        filter_levels = sorted(results_by_filter_level.keys())
        analysis = {
            "filter_levels": filter_levels,
            "val_losses": [results_by_filter_level[f].val_loss for f in filter_levels],
            "downstream_scores": [
                results_by_filter_level[f].downstream for f in filter_levels
            ],
        }

        # Find crossover: where does less-filtered beat more-filtered?
        losses = analysis["val_losses"]
        if len(losses) >= 2:
            # If loss decreases with more data (less filtering), Bitter Lesson holds
            analysis["loss_trend"] = "bitter_lesson" if losses[-1] < losses[0] else "filtering_helps"

        return analysis
