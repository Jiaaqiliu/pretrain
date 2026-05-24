"""Reward policy for pretraining recipe search.

Maps evaluation results to scalar rewards that MCGS backpropagates.
The reward design encodes our research objectives:
- Primary: minimize validation loss (proxy for model quality)
- Secondary: maximize downstream reasoning performance
- Bonus: curriculum novelty (encourages exploring phase transitions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .eval_harness import EvalResult
from .mixture import CurriculumSchedule


@dataclass
class PretrainRewardPolicy:
    """Converts EvalResult + curriculum info into a MCGS-compatible reward.

    Reward is in [0, 1] range for stable backpropagation.
    """

    # Best known loss so far (used for normalization)
    best_loss: float = 4.0
    worst_loss: float = 6.0

    # Whether to reward curriculum novelty
    curriculum_bonus: float = 0.05

    # Downstream weight (only used in full eval mode)
    downstream_weight: float = 0.3

    def compute_reward(
        self,
        eval_result: EvalResult,
        curriculum: CurriculumSchedule,
        parent_eval: EvalResult | None = None,
    ) -> float:
        """Compute reward from eval result.

        Returns:
            Reward in [0, 1]. Higher is better.
        """
        # 1. Loss-based reward (normalized to [0, 1])
        loss_reward = self._loss_to_reward(eval_result.val_loss)

        # 2. Improvement bonus (relative to parent)
        improvement_bonus = 0.0
        if parent_eval is not None and parent_eval.val_loss > 0:
            relative_improvement = (parent_eval.val_loss - eval_result.val_loss) / parent_eval.val_loss
            improvement_bonus = max(0.0, min(0.2, relative_improvement * 2.0))

        # 3. Downstream bonus (if available)
        downstream_bonus = 0.0
        if eval_result.downstream:
            avg_downstream = sum(eval_result.downstream.values()) / len(eval_result.downstream)
            downstream_bonus = avg_downstream * self.downstream_weight

        # 4. Curriculum novelty bonus (small reward for trying curricula)
        novelty_bonus = 0.0
        if not curriculum.is_static:
            novelty_bonus = self.curriculum_bonus

        # Combine
        reward = loss_reward + improvement_bonus + downstream_bonus + novelty_bonus

        # Clip to [0, 1]
        return max(0.0, min(1.0, reward))

    def _loss_to_reward(self, loss: float) -> float:
        """Monotonically decreasing function mapping loss to reward.

        Uses a sigmoid-like transform centered on the expected loss range.
        """
        # Normalize: best_loss → ~0.8, worst_loss → ~0.2
        normalized = (self.worst_loss - loss) / (self.worst_loss - self.best_loss)
        return max(0.0, min(0.9, normalized * 0.7 + 0.1))

    def update_bounds(self, loss: float):
        """Update best/worst loss bounds based on observations."""
        if loss < self.best_loss:
            self.best_loss = loss
        if loss > self.worst_loss:
            self.worst_loss = loss
