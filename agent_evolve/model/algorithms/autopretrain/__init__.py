"""AutoPretrain — Automated pretraining recipe search via proxy-scale MCGS.

This module implements the data mixture search algorithm described in:
"Pretraining Data Mixtures as a Search Problem: From Proxy Models to Kaggle Gold"

Architecture:
    ProxySearchLoop → MixtureMutator → OLMoCoreBackend.run_trial → EvalHarness → RewardSignal
                  ↑                                                                    ↓
                  └──────────────── MCGS backpropagation ←─────────────────────────────┘

Key components:
    - DataMixture: continuous simplex representation of domain ratios
    - MixtureMutator: LLM-guided + structured mutations on the simplex
    - CurriculumSchedule: phase-based data mix transitions during training
    - FilterAggressiveness: per-domain quality threshold (Bitter Lesson axis)
    - ProxySearchLoop: orchestrates MCGS search at 190M scale
    - TransferVerifier: validates proxy findings at 3B scale
"""

from .mixture import DataMixture, CurriculumSchedule, FilterConfig
from .mutator import MixtureMutator
from .search import ProxySearchLoop
from .eval_harness import PretrainEvalHarness
from .reward import PretrainRewardPolicy

__all__ = [
    "DataMixture",
    "CurriculumSchedule",
    "FilterConfig",
    "MixtureMutator",
    "ProxySearchLoop",
    "PretrainEvalHarness",
    "PretrainRewardPolicy",
]
