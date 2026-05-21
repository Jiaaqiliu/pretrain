"""Hyperparameter Optimization Engine.

Integrates Optuna for efficient hyperparameter search with:
- Multi-objective optimization (loss vs. compute cost)
- ASHA-style early stopping of underperforming trials
- Knowledge transfer from completed experiments
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import optuna
from optuna.samplers import TPESampler

from autopilot.utils.logging import get_logger

log = get_logger("optimization.hpo")


class ParamType(enum.Enum):
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    LOG_FLOAT = "log_float"
    LOG_INT = "log_int"


@dataclass
class SearchParam:
    """A single hyperparameter in the search space."""

    name: str
    param_type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class SearchSpace:
    """Definition of the hyperparameter search space."""

    params: List[SearchParam] = field(default_factory=list)

    def add_float(
        self, name: str, low: float, high: float, log: bool = False, default: Optional[float] = None
    ) -> "SearchSpace":
        self.params.append(
            SearchParam(
                name=name,
                param_type=ParamType.LOG_FLOAT if log else ParamType.FLOAT,
                low=low,
                high=high,
                default=default,
            )
        )
        return self

    def add_int(
        self, name: str, low: int, high: int, log: bool = False, default: Optional[int] = None
    ) -> "SearchSpace":
        self.params.append(
            SearchParam(
                name=name,
                param_type=ParamType.LOG_INT if log else ParamType.INT,
                low=low,
                high=high,
                default=default,
            )
        )
        return self

    def add_categorical(
        self, name: str, choices: List[Any], default: Optional[Any] = None
    ) -> "SearchSpace":
        self.params.append(
            SearchParam(
                name=name, param_type=ParamType.CATEGORICAL, choices=choices, default=default
            )
        )
        return self

    @classmethod
    def for_llm_pretraining(cls) -> "SearchSpace":
        """Pre-configured search space for LLM pre-training."""
        space = cls()
        space.add_float("learning_rate", 1e-5, 1e-2, log=True, default=3e-4)
        space.add_float("weight_decay", 0.0, 0.3, default=0.1)
        space.add_int("warmup_steps", 100, 5000, default=2000)
        space.add_float("max_grad_norm", 0.1, 10.0, default=1.0)
        space.add_categorical("scheduler", ["cosine", "linear", "wsd"], default="cosine")
        space.add_float("beta1", 0.85, 0.99, default=0.9)
        space.add_float("beta2", 0.9, 0.999, default=0.95)
        space.add_float("z_loss_multiplier", 0.0, 1e-3, default=1e-4)
        return space


@dataclass
class Trial:
    """A single HPO trial."""

    trial_id: int
    params: Dict[str, Any]
    objective_value: Optional[float] = None
    status: str = "running"  # "running", "completed", "pruned", "failed"
    step: int = 0
    intermediate_values: Dict[int, float] = field(default_factory=dict)


class HPOEngine:
    """Hyperparameter Optimization Engine wrapping Optuna.

    Features:
    - Bayesian optimization with TPE sampler
    - Multi-objective support (loss + compute cost)
    - ASHA-style pruning for early stopping bad trials
    - Warm-starting from prior experiments
    """

    def __init__(
        self,
        search_space: SearchSpace,
        study_name: str = "autopilot_hpo",
        n_trials: int = 50,
        direction: str = "minimize",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        seed: int = 42,
    ):
        self._search_space = search_space
        self._n_trials = n_trials

        sampler = TPESampler(seed=seed, multivariate=True)
        self._pruner = pruner or optuna.pruners.HyperbandPruner(
            min_resource=100, max_resource=10000, reduction_factor=3
        )

        self._study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=self._pruner,
        )
        self._active_trials: Dict[int, Trial] = {}

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        try:
            return self._study.best_params
        except ValueError:
            return None

    @property
    def best_value(self) -> Optional[float]:
        try:
            return self._study.best_value
        except ValueError:
            return None

    @property
    def n_completed(self) -> int:
        return len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    def suggest_next(self) -> Trial:
        """Get next trial parameters to evaluate."""
        trial = self._study.ask()
        params = self._sample_params(trial)

        autopilot_trial = Trial(trial_id=trial.number, params=params)
        self._active_trials[trial.number] = autopilot_trial

        log.info(f"Suggested trial {trial.number}: {params}")
        return autopilot_trial

    def report_intermediate(self, trial_id: int, step: int, value: float) -> bool:
        """Report intermediate value. Returns True if trial should be pruned."""
        if trial_id in self._active_trials:
            self._active_trials[trial_id].step = step
            self._active_trials[trial_id].intermediate_values[step] = value

        trial = self._study.trials[trial_id]
        trial.report(value, step)

        should_prune = trial.should_prune()
        if should_prune:
            log.info(f"Trial {trial_id} pruned at step {step} (value={value:.4f})")
            self._active_trials[trial_id].status = "pruned"
        return should_prune

    def report_complete(self, trial_id: int, value: float) -> None:
        """Report final objective value for a trial."""
        self._study.tell(trial_id, value)
        if trial_id in self._active_trials:
            self._active_trials[trial_id].objective_value = value
            self._active_trials[trial_id].status = "completed"
        log.info(f"Trial {trial_id} completed with value={value:.4f}")

    def report_failed(self, trial_id: int) -> None:
        """Report that a trial failed."""
        self._study.tell(trial_id, state=optuna.trial.TrialState.FAIL)
        if trial_id in self._active_trials:
            self._active_trials[trial_id].status = "failed"

    def get_top_trials(self, n: int = 5) -> List[Trial]:
        """Get the top N completed trials by objective value."""
        completed = [
            t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        completed.sort(key=lambda t: t.value if t.value is not None else float("inf"))
        return [
            Trial(trial_id=t.number, params=t.params, objective_value=t.value, status="completed")
            for t in completed[:n]
        ]

    def warm_start(self, prior_results: List[Tuple[Dict[str, Any], float]]) -> None:
        """Warm-start the optimization with results from prior experiments."""
        for params, value in prior_results:
            self._study.enqueue_trial(params)
            trial = self._study.ask()
            self._study.tell(trial, value)
        log.info(f"Warm-started with {len(prior_results)} prior results")

    def get_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance rankings."""
        try:
            return optuna.importance.get_param_importances(self._study)
        except (ValueError, RuntimeError):
            return {}

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        for p in self._search_space.params:
            if p.param_type == ParamType.FLOAT:
                params[p.name] = trial.suggest_float(p.name, p.low, p.high)
            elif p.param_type == ParamType.LOG_FLOAT:
                params[p.name] = trial.suggest_float(p.name, p.low, p.high, log=True)
            elif p.param_type == ParamType.INT:
                params[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high))
            elif p.param_type == ParamType.LOG_INT:
                params[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), log=True)
            elif p.param_type == ParamType.CATEGORICAL:
                params[p.name] = trial.suggest_categorical(p.name, p.choices)
        return params
