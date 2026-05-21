"""Training analysis agent.

Analyzes experimental results to:
- Identify key factors driving performance
- Compare experiments and extract insights
- Diagnose training issues
- Generate human-readable reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


from autopilot.monitoring.anomaly import Severity, TrainingAnomaly
from autopilot.monitoring.metrics import MetricsCollector
from autopilot.optimization.hpo import HPOEngine
from autopilot.utils.logging import get_logger

log = get_logger("agent.analyzer")


@dataclass
class ExperimentInsight:
    """A single insight derived from experiment analysis."""

    category: str  # "performance", "stability", "efficiency", "comparison"
    title: str
    description: str
    confidence: float  # 0-1
    actionable: bool = False
    suggested_action: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Complete analysis report for a set of experiments."""

    summary: str
    insights: List[ExperimentInsight]
    rankings: List[Tuple[str, float]]  # (experiment_id, loss)
    key_factors: Dict[str, float]  # param_name -> importance
    recommendations: List[str]
    raw_data: Dict[str, Any] = field(default_factory=dict)


class AnalyzerAgent:
    """Analyzes training experiments to extract actionable insights.

    Capabilities:
    - Factor importance analysis (which HPs matter most)
    - Cross-experiment comparison
    - Convergence efficiency analysis
    - Stability assessment
    - Recommendation generation
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self._metrics = metrics_collector

    def analyze_sweep(
        self,
        experiment_configs: Dict[str, Dict[str, Any]],
        hpo_engine: Optional[HPOEngine] = None,
    ) -> AnalysisReport:
        """Analyze results from a hyperparameter sweep."""
        insights: List[ExperimentInsight] = []

        # Get rankings
        rankings = self._compute_rankings(list(experiment_configs.keys()))

        # Get factor importance
        key_factors = {}
        if hpo_engine:
            key_factors = hpo_engine.get_importance()
            if key_factors:
                top_factor = max(key_factors, key=key_factors.get)
                insights.append(
                    ExperimentInsight(
                        category="performance",
                        title=f"Most important parameter: {top_factor}",
                        description=(
                            f"Parameter '{top_factor}' has importance "
                            f"{key_factors[top_factor]:.3f}, explaining the most "
                            f"variance in training outcomes."
                        ),
                        confidence=0.8,
                        actionable=True,
                        suggested_action=f"Focus tuning on {top_factor}",
                        evidence={"importance_scores": key_factors},
                    )
                )

        # Convergence analysis
        convergence_insights = self._analyze_convergence(experiment_configs)
        insights.extend(convergence_insights)

        # Stability analysis
        stability_insights = self._analyze_stability(list(experiment_configs.keys()))
        insights.extend(stability_insights)

        # Generate recommendations
        recommendations = self._generate_recommendations(insights, rankings, key_factors)

        # Summary
        best_id = rankings[0][0] if rankings else "none"
        best_loss = rankings[0][1] if rankings else 0.0
        summary = (
            f"Analyzed {len(experiment_configs)} experiments. "
            f"Best: {best_id} (loss={best_loss:.4f}). "
            f"Key factor: {max(key_factors, key=key_factors.get) if key_factors else 'N/A'}."
        )

        return AnalysisReport(
            summary=summary,
            insights=insights,
            rankings=rankings,
            key_factors=key_factors,
            recommendations=recommendations,
        )

    def compare_experiments(
        self, experiment_ids: List[str], configs: Dict[str, Dict[str, Any]]
    ) -> List[ExperimentInsight]:
        """Compare specific experiments and identify what differs."""
        insights = []

        if len(experiment_ids) < 2:
            return insights

        comparison = self._metrics.compare_experiments(experiment_ids, "loss")

        # Find best and worst
        sorted_exps = sorted(
            comparison.items(),
            key=lambda x: x[1].get("current") or float("inf"),
        )

        if len(sorted_exps) >= 2:
            best_id, best_data = sorted_exps[0]
            worst_id, worst_data = sorted_exps[-1]

            best_config = configs.get(best_id, {})
            worst_config = configs.get(worst_id, {})

            # Find differing parameters
            diffs = self._find_config_diffs(best_config, worst_config)
            if diffs:
                insights.append(
                    ExperimentInsight(
                        category="comparison",
                        title="Key differences between best and worst",
                        description=(
                            f"Best ({best_id}, loss={best_data.get('current'):.4f}) vs "
                            f"Worst ({worst_id}, loss={worst_data.get('current'):.4f}). "
                            f"Differing params: {', '.join(diffs.keys())}"
                        ),
                        confidence=0.7,
                        evidence={"diffs": diffs, "best": best_id, "worst": worst_id},
                    )
                )

        return insights

    def diagnose_anomaly(
        self, experiment_id: str, anomaly: TrainingAnomaly, config: Dict[str, Any]
    ) -> ExperimentInsight:
        """Diagnose a training anomaly and suggest fixes."""
        self._metrics.get_window(experiment_id)

        diagnosis = f"Anomaly: {anomaly.message}"
        suggestion = anomaly.suggested_action

        # Enhanced diagnosis based on anomaly type and context
        if anomaly.anomaly_type.value == "loss_spike":
            lr = config.get("optimizer", {}).get("lr", "unknown")
            diagnosis += (
                f" Current LR: {lr}. "
                f"Consider reducing learning rate or increasing gradient clipping."
            )
            if anomaly.severity == Severity.CRITICAL:
                suggestion = "rollback_and_reduce_lr"
            else:
                suggestion = "skip_step_and_monitor"

        elif anomaly.anomaly_type.value == "slow_convergence":
            diagnosis += " Training may have reached a plateau or learning rate is too low."
            suggestion = "increase_lr_or_early_stop"

        elif anomaly.anomaly_type.value == "gradient_explosion":
            grad_clip = config.get("optimizer", {}).get("max_grad_norm", "unknown")
            diagnosis += f" Current grad clip: {grad_clip}. Consider tighter clipping."
            suggestion = "reduce_grad_clip"

        return ExperimentInsight(
            category="stability",
            title=f"Diagnosis: {anomaly.anomaly_type.value}",
            description=diagnosis,
            confidence=0.6,
            actionable=True,
            suggested_action=suggestion,
            evidence={"anomaly": anomaly.__dict__, "config": config},
        )

    def _compute_rankings(self, experiment_ids: List[str]) -> List[Tuple[str, float]]:
        comparison = self._metrics.compare_experiments(experiment_ids, "loss")
        rankings = []
        for eid, data in comparison.items():
            current = data.get("current")
            if current is not None:
                rankings.append((eid, current))
        rankings.sort(key=lambda x: x[1])
        return rankings

    def _analyze_convergence(
        self, experiment_configs: Dict[str, Dict[str, Any]]
    ) -> List[ExperimentInsight]:
        insights = []

        for eid in experiment_configs:
            window = self._metrics.get_window(eid)
            if window is None or window.length < 50:
                continue

            trend = window.trend("loss")
            if trend is not None and trend > 0:
                insights.append(
                    ExperimentInsight(
                        category="performance",
                        title=f"Diverging: {eid}",
                        description=f"Experiment {eid} shows positive loss trend ({trend:.6f}/step)",
                        confidence=0.7,
                        actionable=True,
                        suggested_action="early_stop",
                    )
                )

        return insights

    def _analyze_stability(self, experiment_ids: List[str]) -> List[ExperimentInsight]:
        insights = []

        for eid in experiment_ids:
            window = self._metrics.get_window(eid)
            if window is None or window.length < 50:
                continue

            loss_std = window.std("loss")
            loss_mean = window.mean("loss")
            if loss_std and loss_mean and loss_mean > 0:
                cv = loss_std / loss_mean  # coefficient of variation
                if cv > 0.1:
                    insights.append(
                        ExperimentInsight(
                            category="stability",
                            title=f"High variance: {eid}",
                            description=(
                                f"Loss CV={cv:.3f} (std={loss_std:.4f}, mean={loss_mean:.4f})"
                            ),
                            confidence=0.6,
                        )
                    )

        return insights

    def _find_config_diffs(
        self, config_a: Dict[str, Any], config_b: Dict[str, Any]
    ) -> Dict[str, Tuple[Any, Any]]:
        """Find parameters that differ between two configs."""
        diffs = {}
        all_keys = set(config_a.keys()) | set(config_b.keys())
        for key in all_keys:
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            if val_a != val_b:
                if isinstance(val_a, dict) and isinstance(val_b, dict):
                    sub_diffs = self._find_config_diffs(val_a, val_b)
                    for sub_key, (sa, sb) in sub_diffs.items():
                        diffs[f"{key}.{sub_key}"] = (sa, sb)
                else:
                    diffs[key] = (val_a, val_b)
        return diffs

    def _generate_recommendations(
        self,
        insights: List[ExperimentInsight],
        rankings: List[Tuple[str, float]],
        key_factors: Dict[str, float],
    ) -> List[str]:
        recommendations = []

        # Recommendation based on key factors
        if key_factors:
            top_params = sorted(key_factors, key=key_factors.get, reverse=True)[:3]
            recommendations.append(
                f"Focus tuning on: {', '.join(top_params)} (highest importance)"
            )

        # Recommendation based on stability
        unstable = [i for i in insights if i.category == "stability" and i.confidence > 0.5]
        if unstable:
            recommendations.append(
                f"{len(unstable)} experiments show instability. "
                "Consider reducing learning rate or increasing warmup."
            )

        # Recommendation based on convergence
        diverging = [
            i for i in insights if i.category == "performance" and "Diverging" in i.title
        ]
        if diverging:
            recommendations.append(
                f"{len(diverging)} experiments are diverging. Stop them and reallocate resources."
            )

        if not recommendations:
            recommendations.append("Training appears healthy. Continue monitoring.")

        return recommendations
