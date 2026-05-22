"""Thermodynamic analysis: fitting state equations, KWW relaxation, correlations.

Implements the analysis needed for each research question:
  Q1: State equation fitting (P·V = N·k_eff·T + α·N^(2/3)·T^(3/2) - γ/V)
  Q2: WSD vs Cosine entropy production comparison
  Q3: KWW stretched exponential fitting
  Q4: Order parameter correlation with downstream benchmarks
  Q5: Schedule comparison statistical tests
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, ttest_rel


# =============================================================================
# Q1: State Equation Fitting
# =============================================================================


@dataclass
class StateEquationFit:
    """Result of fitting the modified state equation."""

    k0: float
    alpha: float  # finite-size correction
    gamma: float  # attractive interaction term
    k_eff: float  # k0 + alpha * N^(-1/3)
    r_squared: float
    residuals: np.ndarray
    params_cov: np.ndarray
    model_size: str = ""
    num_points: int = 0


def ideal_gas_ratio(P: np.ndarray, V: np.ndarray, N: int, T: np.ndarray) -> np.ndarray:
    """Compute P·V / (N·T) — should converge to k_eff during stable phase."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (P * V) / (N * T)
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def fit_k_eff(
    pv_over_nt_values: np.ndarray,
    model_sizes_N: np.ndarray,
) -> tuple[float, float]:
    """Fit k_eff(N) = k₀ + α·N^(-1/3) across multiple scales.

    Args:
        pv_over_nt_values: converged k_eff values for each scale
        model_sizes_N: corresponding N (number of parameters)

    Returns:
        (k0, alpha) fitted parameters
    """
    def model(N, k0, alpha):
        return k0 + alpha * np.power(N.astype(float), -1.0 / 3.0)

    popt, pcov = curve_fit(model, model_sizes_N, pv_over_nt_values, p0=[0.5, 1.0])
    return popt[0], popt[1]


def fit_state_equation(
    P: np.ndarray,
    V: np.ndarray,
    N: int,
    T: np.ndarray,
) -> StateEquationFit:
    """Fit the modified state equation: P·V = N·k_eff·T + α·N^(2/3)·T^(3/2) - γ/V.

    Args:
        P: pressure (weight decay) array
        V: volume (||θ||²_F) array
        N: number of parameters (scalar)
        T: temperature array
    """
    PV = P * V

    def model(X, k_eff, alpha, gamma):
        T_arr, V_arr = X
        return (
            N * k_eff * T_arr
            + alpha * N ** (2.0 / 3.0) * T_arr ** 1.5
            - gamma / V_arr
        )

    mask = np.isfinite(PV) & np.isfinite(T) & (T > 0) & (V > 0)
    T_valid = T[mask]
    V_valid = V[mask]
    PV_valid = PV[mask]

    if len(T_valid) < 5:
        return StateEquationFit(
            k0=0.0, alpha=0.0, gamma=0.0, k_eff=0.0,
            r_squared=0.0, residuals=np.array([]),
            params_cov=np.zeros((3, 3)), num_points=0,
        )

    try:
        popt, pcov = curve_fit(
            model,
            (T_valid, V_valid),
            PV_valid,
            p0=[0.5, 0.01, 0.1],
            maxfev=10000,
        )
    except RuntimeError:
        return StateEquationFit(
            k0=0.0, alpha=0.0, gamma=0.0, k_eff=0.0,
            r_squared=0.0, residuals=np.array([]),
            params_cov=np.zeros((3, 3)), num_points=len(T_valid),
        )

    predicted = model((T_valid, V_valid), *popt)
    residuals = PV_valid - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((PV_valid - PV_valid.mean()) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)

    return StateEquationFit(
        k0=popt[0],
        alpha=popt[1],
        gamma=popt[2],
        k_eff=popt[0],
        r_squared=r_squared,
        residuals=residuals,
        params_cov=pcov,
        num_points=len(T_valid),
    )


def identify_thermodynamic_regime(
    T: np.ndarray, pv_over_nt: np.ndarray, threshold_high: float = 0.9, threshold_low: float = 0.3
) -> np.ndarray:
    """Classify each step into thermodynamic regime: ideal_gas / liquid / glass.

    Based on how close P·V/(N·T) is to the ideal value k_eff.
    """
    # Estimate k_eff as the median during stable phase
    stable_mask = np.isfinite(pv_over_nt)
    if not np.any(stable_mask):
        return np.full(len(T), "unknown")

    k_eff_est = np.nanmedian(pv_over_nt[stable_mask])

    regimes = np.full(len(T), "liquid", dtype="<U10")
    if k_eff_est > 0:
        deviation = np.abs(pv_over_nt - k_eff_est) / k_eff_est
        regimes[deviation < 0.05] = "ideal_gas"
        regimes[deviation > 0.3] = "glass"

    return regimes


# =============================================================================
# Q2: WSD vs Cosine Entropy Comparison
# =============================================================================


@dataclass
class EntropyComparison:
    """Comparison of cumulative entropy production between two schedules."""

    schedule_a: str
    schedule_b: str
    delta_s_a: float  # cumulative entropy production of schedule A
    delta_s_b: float  # cumulative entropy production of schedule B
    reduction_pct: float  # percentage reduction: (B - A) / B * 100
    efficiency_a: float  # thermodynamic efficiency of A
    efficiency_b: float  # thermodynamic efficiency of B
    model_size: str = ""


def compare_entropy_production(
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    steps_a: np.ndarray,
    steps_b: np.ndarray,
    delta_f_a: float,
    delta_f_b: float,
    mean_t_a: float,
    mean_t_b: float,
    schedule_a: str = "wsd",
    schedule_b: str = "cosine",
) -> EntropyComparison:
    """Compare entropy production between two schedules at matched compute."""
    from experiments.thermodynamics.measures import (
        cumulative_entropy_production,
        thermodynamic_efficiency,
    )

    cum_a = cumulative_entropy_production(sigma_a, steps_a)
    cum_b = cumulative_entropy_production(sigma_b, steps_b)

    total_a = cum_a[-1] if len(cum_a) > 0 else 0.0
    total_b = cum_b[-1] if len(cum_b) > 0 else 0.0

    reduction = (total_b - total_a) / max(total_b, 1e-15) * 100

    eff_a = thermodynamic_efficiency(delta_f_a, total_a, mean_t_a)
    eff_b = thermodynamic_efficiency(delta_f_b, total_b, mean_t_b)

    return EntropyComparison(
        schedule_a=schedule_a,
        schedule_b=schedule_b,
        delta_s_a=total_a,
        delta_s_b=total_b,
        reduction_pct=reduction,
        efficiency_a=eff_a,
        efficiency_b=eff_b,
    )


# =============================================================================
# Q3: KWW Glass Relaxation Fitting
# =============================================================================


@dataclass
class KWWFit:
    """Result of KWW stretched exponential fit."""

    tau: float  # characteristic relaxation time (in steps or tokens)
    beta: float  # stretching exponent (0.5-0.8 = glassy)
    s_inf: float  # asymptotic spectral entropy
    s_0: float  # initial spectral entropy at onset
    r_squared: float
    tau_ci: tuple[float, float] = (0.0, 0.0)  # 95% CI for tau
    beta_ci: tuple[float, float] = (0.0, 0.0)  # 95% CI for beta
    bic_kww: float = 0.0
    bic_exponential: float = 0.0
    bic_powerlaw: float = 0.0
    optimal_duration_3tau: float = 0.0  # 3τ tokens for 95% relaxation


def kww_function(t: np.ndarray, tau: float, beta: float) -> np.ndarray:
    """KWW stretched exponential: φ(t) = exp[-(t/τ)^β]."""
    return np.exp(-np.power(t / tau, beta))


def exponential_function(t: np.ndarray, tau: float) -> np.ndarray:
    """Simple exponential: φ(t) = exp(-t/τ)."""
    return np.exp(-t / tau)


def powerlaw_function(t: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    """Power law: φ(t) = (1 + t/τ)^(-α)."""
    return np.power(1.0 + t / tau, -alpha)


def _bic(n: int, k: int, sse: float) -> float:
    """Bayesian Information Criterion."""
    if sse <= 0 or n <= k:
        return np.inf
    return n * np.log(sse / n) + k * np.log(n)


def fit_kww(
    spectral_entropy: np.ndarray,
    steps: np.ndarray,
    onset_step: int,
    n_bootstrap: int = 1000,
) -> KWWFit:
    """Fit KWW stretched exponential to mid-training spectral entropy relaxation.

    Args:
        spectral_entropy: S(t) values
        steps: corresponding step numbers
        onset_step: step at which mid-training begins
        n_bootstrap: number of bootstrap samples for CIs

    Returns:
        KWWFit with parameters, R², confidence intervals, and BIC comparison
    """
    # Select post-onset data
    mask = steps >= onset_step
    s_data = spectral_entropy[mask]
    t_data = (steps[mask] - onset_step).astype(float)

    if len(s_data) < 10:
        return KWWFit(tau=0.0, beta=0.0, s_inf=0.0, s_0=0.0, r_squared=0.0)

    # Estimate S_0 and S_inf
    s_0 = s_data[0]
    # S_inf from EMA of last 10%
    last_10pct = s_data[int(0.9 * len(s_data)):]
    s_inf = float(np.mean(last_10pct))

    # Normalize to φ(t) = (S(t) - S_inf) / (S_0 - S_inf)
    denom = s_0 - s_inf
    if abs(denom) < 1e-10:
        return KWWFit(tau=0.0, beta=0.0, s_inf=s_inf, s_0=s_0, r_squared=0.0)

    phi = (s_data - s_inf) / denom
    phi = np.clip(phi, 1e-10, 1.0)

    # Fit KWW
    try:
        popt_kww, pcov_kww = curve_fit(
            kww_function, t_data, phi,
            p0=[t_data[-1] / 3, 0.6],
            bounds=([1.0, 0.1], [t_data[-1] * 10, 1.0]),
            maxfev=10000,
        )
        tau_fit, beta_fit = popt_kww
    except RuntimeError:
        return KWWFit(tau=0.0, beta=0.0, s_inf=s_inf, s_0=s_0, r_squared=0.0)

    # R² for KWW
    phi_pred = kww_function(t_data, tau_fit, beta_fit)
    ss_res = np.sum((phi - phi_pred) ** 2)
    ss_tot = np.sum((phi - phi.mean()) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)

    # BIC comparison
    n = len(phi)
    bic_kww = _bic(n, 2, ss_res)

    # Fit simple exponential
    try:
        popt_exp, _ = curve_fit(
            exponential_function, t_data, phi,
            p0=[t_data[-1] / 3],
            bounds=([1.0], [t_data[-1] * 10]),
        )
        ss_res_exp = np.sum((phi - exponential_function(t_data, *popt_exp)) ** 2)
        bic_exp = _bic(n, 1, ss_res_exp)
    except RuntimeError:
        bic_exp = np.inf

    # Fit power law
    try:
        popt_pl, _ = curve_fit(
            powerlaw_function, t_data, phi,
            p0=[t_data[-1] / 3, 1.0],
            bounds=([1.0, 0.1], [t_data[-1] * 10, 10.0]),
        )
        ss_res_pl = np.sum((phi - powerlaw_function(t_data, *popt_pl)) ** 2)
        bic_pl = _bic(n, 2, ss_res_pl)
    except RuntimeError:
        bic_pl = np.inf

    # Bootstrap confidence intervals
    tau_boots = []
    beta_boots = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        residuals = phi - phi_pred
        boot_residuals = rng.choice(residuals, size=len(residuals), replace=True)
        phi_boot = phi_pred + boot_residuals
        phi_boot = np.clip(phi_boot, 1e-10, 1.0)
        try:
            popt_boot, _ = curve_fit(
                kww_function, t_data, phi_boot,
                p0=[tau_fit, beta_fit],
                bounds=([1.0, 0.1], [t_data[-1] * 10, 1.0]),
                maxfev=5000,
            )
            tau_boots.append(popt_boot[0])
            beta_boots.append(popt_boot[1])
        except RuntimeError:
            continue

    tau_ci = (
        np.percentile(tau_boots, 2.5) if tau_boots else tau_fit,
        np.percentile(tau_boots, 97.5) if tau_boots else tau_fit,
    )
    beta_ci = (
        np.percentile(beta_boots, 2.5) if beta_boots else beta_fit,
        np.percentile(beta_boots, 97.5) if beta_boots else beta_fit,
    )

    return KWWFit(
        tau=tau_fit,
        beta=beta_fit,
        s_inf=s_inf,
        s_0=s_0,
        r_squared=r_squared,
        tau_ci=tau_ci,
        beta_ci=beta_ci,
        bic_kww=bic_kww,
        bic_exponential=bic_exp,
        bic_powerlaw=bic_pl,
        optimal_duration_3tau=3 * tau_fit,
    )


# =============================================================================
# Q4: Order Parameter Correlation with Downstream Benchmarks
# =============================================================================


@dataclass
class MonitorCorrelation:
    """Correlation of a training metric with downstream benchmark performance."""

    metric_name: str
    spearman_r: float
    p_value: float
    per_benchmark: dict = field(default_factory=dict)


def compute_monitoring_correlations(
    steps: np.ndarray,
    metrics: dict[str, np.ndarray],
    benchmark_scores: dict[str, np.ndarray],
) -> list[MonitorCorrelation]:
    """Compute Spearman correlations between training metrics and downstream benchmarks.

    Args:
        steps: array of checkpoint steps
        metrics: dict of metric_name → values at each step
            (e.g., {"loss": [...], "spectral_entropy": [...], "order_parameter": [...], "free_energy": [...]})
        benchmark_scores: dict of benchmark_name → scores at each step
            (e.g., {"mmlu": [...], "gsm8k": [...], ...})

    Returns:
        list of MonitorCorrelation, sorted by average Spearman r (descending)
    """
    # Average benchmark score across all benchmarks at each step
    all_scores = np.stack(list(benchmark_scores.values()), axis=0)
    avg_score = all_scores.mean(axis=0)

    results = []
    for metric_name, metric_values in metrics.items():
        # Overall correlation with average score
        valid = np.isfinite(metric_values) & np.isfinite(avg_score)
        if valid.sum() < 5:
            continue

        r, p = spearmanr(metric_values[valid], avg_score[valid])

        # Per-benchmark correlations
        per_bench = {}
        for bench_name, bench_scores in benchmark_scores.items():
            valid_b = np.isfinite(metric_values) & np.isfinite(bench_scores)
            if valid_b.sum() >= 5:
                r_b, _ = spearmanr(metric_values[valid_b], bench_scores[valid_b])
                per_bench[bench_name] = float(r_b)

        # For metrics where lower is better (loss, entropy), negate correlation
        if metric_name in ("loss", "internal_energy", "spectral_entropy"):
            r = -r
            per_bench = {k: -v for k, v in per_bench.items()}

        results.append(MonitorCorrelation(
            metric_name=metric_name,
            spearman_r=abs(float(r)),
            p_value=float(p),
            per_benchmark=per_bench,
        ))

    return sorted(results, key=lambda x: x.spearman_r, reverse=True)


# =============================================================================
# Q5: Schedule Comparison Statistical Tests
# =============================================================================


@dataclass
class ScheduleComparison:
    """Statistical comparison of training schedules."""

    schedule: str
    baseline: str
    mean_loss: float
    baseline_mean_loss: float
    delta_pct: float  # (schedule - baseline) / baseline * 100
    std_error: float
    p_value: float
    significant: bool  # p < 0.05
    efficiency: float
    baseline_efficiency: float


def compare_schedules(
    results: dict[str, list[float]],
    baseline: str = "wsd_linear",
) -> list[ScheduleComparison]:
    """Compare schedules against baseline using paired t-test.

    Args:
        results: dict of schedule_name → list of final losses (one per seed)
        baseline: name of baseline schedule

    Returns:
        list of ScheduleComparison (sorted by delta_pct)
    """
    if baseline not in results:
        raise ValueError(f"Baseline '{baseline}' not in results")

    baseline_losses = np.array(results[baseline])
    baseline_mean = baseline_losses.mean()
    comparisons = []

    for schedule, losses in results.items():
        if schedule == baseline:
            continue

        losses_arr = np.array(losses)
        mean_loss = losses_arr.mean()
        delta_pct = (mean_loss - baseline_mean) / baseline_mean * 100
        std_error = losses_arr.std() / np.sqrt(len(losses_arr))

        # Paired t-test
        if len(losses_arr) == len(baseline_losses) and len(losses_arr) >= 2:
            _, p_value = ttest_rel(losses_arr, baseline_losses)
        else:
            p_value = 1.0

        comparisons.append(ScheduleComparison(
            schedule=schedule,
            baseline=baseline,
            mean_loss=float(mean_loss),
            baseline_mean_loss=float(baseline_mean),
            delta_pct=float(delta_pct),
            std_error=float(std_error),
            p_value=float(p_value),
            significant=p_value < 0.05,
            efficiency=0.0,
            baseline_efficiency=0.0,
        ))

    return sorted(comparisons, key=lambda x: x.delta_pct)
