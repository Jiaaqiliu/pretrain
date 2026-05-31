"""MoE-specific analysis: hypothesis testing, phase detection, reversal detection.

Implements analysis for all 22 hypotheses (M1-M6, U1-U5, N1-N11).
Reads JSONL output from moe_measures.py and produces analysis results.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_moe_trajectory(jsonl_path: str) -> list[dict]:
    """Load checkpoint measurements sorted by step."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("step", 0))
    return records


# ---------------------------------------------------------------------------
# M1: SR/d convergence per expert
# ---------------------------------------------------------------------------

@dataclass
class SrdConvergenceResult:
    model: str
    srd_final: float
    srd_init: float
    delta_h2: float  # log(srd_final/srd_init) ≈ ΔH₂
    convergence_step: int
    in_dense_range: bool  # 0.04-0.07


def analyze_srd_convergence(trajectory: list[dict], dense_range=(0.04, 0.07)) -> SrdConvergenceResult:
    """M1: Check if MoE SR/d converges to the dense model range."""
    if not trajectory:
        return None
    srds = [r["srd_moe"] for r in trajectory if r.get("srd_moe")]
    steps = [r["step"] for r in trajectory if r.get("srd_moe")]
    if len(srds) < 2:
        return None

    srd_init = srds[0]
    srd_final = srds[-1]
    delta_h2 = np.log(srd_final / srd_init) if srd_init > 0 else 0

    # Find convergence step (when srd stops changing by >5%)
    conv_step = steps[-1]
    for i in range(len(srds) - 1, 0, -1):
        if abs(srds[i] - srds[i-1]) / max(srds[i], 1e-10) > 0.05:
            conv_step = steps[min(i+1, len(steps)-1)]
            break

    return SrdConvergenceResult(
        model=trajectory[0].get("model_name", ""),
        srd_final=srd_final,
        srd_init=srd_init,
        delta_h2=delta_h2,
        convergence_step=conv_step,
        in_dense_range=dense_range[0] <= srd_final <= dense_range[1],
    )


# ---------------------------------------------------------------------------
# M2: Per-expert α reversal detection
# ---------------------------------------------------------------------------

@dataclass
class ReversalEvent:
    step: int
    layer_idx: int
    alpha_at_min: float
    alpha_at_detection: float
    delta_alpha: float


def detect_alpha_reversal(
    trajectory: list[dict],
    patience: int = 3,
    use_per_layer: bool = True,
) -> list[ReversalEvent]:
    """M2: Detect α reversal in MoE training trajectory.

    Checks both global and per-layer α.
    Returns list of reversal events.
    """
    events = []

    # Global reversal
    alphas = [r["alpha_moe"] for r in trajectory if r.get("alpha_moe")]
    steps = [r["step"] for r in trajectory if r.get("alpha_moe")]
    global_events = _detect_reversal_in_series(alphas, steps, patience)
    for step, alpha_min, alpha_det, da in global_events:
        events.append(ReversalEvent(step=step, layer_idx=-1,
                                    alpha_at_min=alpha_min, alpha_at_detection=alpha_det,
                                    delta_alpha=da))

    # Per-layer reversal
    if use_per_layer:
        all_layers = set()
        for r in trajectory:
            for ls in r.get("per_layer_summary", []):
                all_layers.add(ls["layer"])

        for layer_idx in sorted(all_layers):
            layer_alphas = []
            layer_steps = []
            for r in trajectory:
                for ls in r.get("per_layer_summary", []):
                    if ls["layer"] == layer_idx and ls.get("alpha_mean"):
                        layer_alphas.append(ls["alpha_mean"])
                        layer_steps.append(r["step"])
                        break

            layer_events = _detect_reversal_in_series(layer_alphas, layer_steps, patience)
            for step, alpha_min, alpha_det, da in layer_events:
                events.append(ReversalEvent(step=step, layer_idx=layer_idx,
                                            alpha_at_min=alpha_min, alpha_at_detection=alpha_det,
                                            delta_alpha=da))

    return events


def _detect_reversal_in_series(
    values: list[float], steps: list[int], patience: int
) -> list[tuple]:
    """Detect reversal (sustained increase) in a time series."""
    if len(values) < patience + 1:
        return []

    results = []
    min_val = values[0]
    min_step = steps[0]
    increasing_count = 0

    for i in range(1, len(values)):
        if values[i] < min_val:
            min_val = values[i]
            min_step = steps[i]
            increasing_count = 0
        elif values[i] > values[i-1]:
            increasing_count += 1
            if increasing_count >= patience:
                results.append((steps[i], min_val, values[i], values[i] - min_val))
                break
        else:
            increasing_count = 0

    return results


# ---------------------------------------------------------------------------
# M4: Phase transition analysis (total vs active vs per-expert params)
# ---------------------------------------------------------------------------

def sigmoid(x, x0, k, alpha_low, alpha_high):
    return alpha_low + (alpha_high - alpha_low) / (1 + np.exp(-k * (x - x0)))


@dataclass
class PhaseTransitionFit:
    param_type: str  # "total", "active", "per_expert"
    threshold_n: float  # N at transition midpoint
    r_squared: float
    alpha_low: float
    alpha_high: float
    steepness: float


def analyze_phase_transition(
    model_data: list[dict],
) -> list[PhaseTransitionFit]:
    """M4: Fit sigmoid α(N) for three definitions of N.

    model_data: list of dicts with keys:
      total_params, active_params, num_experts, alpha_moe
    """
    results = []

    for param_type in ["total", "active", "per_expert"]:
        ns = []
        alphas = []
        for d in model_data:
            alpha = d.get("alpha_moe", d.get("alpha_mean"))
            if alpha is None or alpha == 0:
                continue

            if param_type == "total":
                n = d["total_params"]
            elif param_type == "active":
                n = d.get("active_params", d["total_params"])
            else:
                ne = d.get("num_experts", 1)
                n = d["total_params"] / max(ne, 1)

            ns.append(np.log10(n))
            alphas.append(alpha)

        if len(ns) < 4:
            continue

        ns = np.array(ns)
        alphas = np.array(alphas)

        try:
            popt, _ = curve_fit(
                sigmoid, ns, alphas,
                p0=[9.2, 10.0, 2.5, 5.0],
                bounds=([6, 0.1, 1, 3], [12, 100, 5, 10]),
                maxfev=5000,
            )
            pred = sigmoid(ns, *popt)
            ss_res = np.sum((alphas - pred) ** 2)
            ss_tot = np.sum((alphas - alphas.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            results.append(PhaseTransitionFit(
                param_type=param_type,
                threshold_n=10 ** popt[0],
                r_squared=r2,
                alpha_low=popt[2],
                alpha_high=popt[3],
                steepness=popt[1],
            ))
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# N1: Expert collapse detection
# ---------------------------------------------------------------------------

@dataclass
class CollapseSignal:
    step: int
    layer_idx: int
    alpha_variance: float  # Var(α) across experts — low = collapse
    alignment: float  # cross-expert alignment — high = collapse
    collapse_score: float  # composite: alignment / (1 + alpha_variance)


def detect_expert_collapse(
    trajectory: list[dict],
    alignment_threshold: float = 0.9,
    alpha_var_threshold: float = 0.1,
) -> list[CollapseSignal]:
    """N1: Detect expert collapse from spectral signals."""
    signals = []

    for r in trajectory:
        for ls in r.get("per_layer_summary", []):
            align = ls.get("alignment", 0)
            alpha_std = ls.get("alpha_std", 0)
            alpha_var = alpha_std ** 2

            if align > alignment_threshold and alpha_var < alpha_var_threshold:
                collapse_score = align / (1 + alpha_var)
                signals.append(CollapseSignal(
                    step=r["step"],
                    layer_idx=ls["layer"],
                    alpha_variance=alpha_var,
                    alignment=align,
                    collapse_score=collapse_score,
                ))

    return signals


# ---------------------------------------------------------------------------
# N3: Energy equipartition analysis
# ---------------------------------------------------------------------------

@dataclass
class EquipartitionResult:
    step: int
    epr_mean: float  # mean EPR across layers
    epr_trend: str  # "increasing" (specializing), "decreasing" (equalizing), "stable"


def analyze_equipartition(trajectory: list[dict]) -> list[EquipartitionResult]:
    """N3: Track energy equipartition ratio across training."""
    results = []
    eprs = []

    for r in trajectory:
        epr = r.get("epr_mean", 0)
        eprs.append(epr)
        results.append(EquipartitionResult(
            step=r["step"],
            epr_mean=epr,
            epr_trend="stable",
        ))

    # Determine trends using sliding window
    if len(eprs) >= 3:
        for i in range(1, len(eprs) - 1):
            if eprs[i] > eprs[i-1] * 1.05:
                results[i].epr_trend = "increasing"
            elif eprs[i] < eprs[i-1] * 0.95:
                results[i].epr_trend = "decreasing"

    return results


# ---------------------------------------------------------------------------
# N5: Three-phase dynamics detection
# ---------------------------------------------------------------------------

@dataclass
class PhaseTransition:
    phase_name: str  # "routing_formation", "specialization", "saturation"
    start_step: int
    end_step: int
    alpha_start: float
    alpha_end: float
    alignment_start: float
    alignment_end: float


def detect_three_phases(
    trajectory: list[dict],
    smoothing_window: int = 5,
) -> list[PhaseTransition]:
    """N5: Detect three dynamic phases in MoE training.

    Phase I:  Routing Formation — alignment drops, α drops fast
    Phase II: Specialization — α variance increases, alignment stable
    Phase III: Saturation — α stable/reversal, alignment frozen
    """
    if len(trajectory) < 6:
        return []

    steps = np.array([r["step"] for r in trajectory])
    alphas = np.array([r.get("alpha_moe", r.get("alpha_mean", 0)) for r in trajectory])
    alignments = np.array([r.get("cross_expert_alignment_mean", 0) for r in trajectory])
    alpha_stds = np.array([r.get("alpha_std_across_experts", 0) for r in trajectory])

    # Smooth
    w = min(smoothing_window, len(alphas))
    if w >= 3 and w % 2 == 0:
        w -= 1
    if w >= 3:
        alphas_s = savgol_filter(alphas, w, min(2, w-1))
        align_s = savgol_filter(alignments, w, min(2, w-1))
    else:
        alphas_s = alphas
        align_s = alignments

    # Compute derivatives
    dalpha = np.gradient(alphas_s)
    dalign = np.gradient(align_s)

    # Phase I ends when alignment stops dropping significantly
    phase1_end = 0
    for i in range(1, len(dalign)):
        if dalign[i] > -0.001 and i > len(dalign) * 0.05:
            phase1_end = i
            break
    if phase1_end == 0:
        phase1_end = len(trajectory) // 4

    # Phase II ends when alpha variance stops increasing or alpha starts reversal
    phase2_end = phase1_end
    for i in range(phase1_end + 1, len(dalpha)):
        if dalpha[i] > 0 and i > phase1_end + 2:
            phase2_end = i
            break
    if phase2_end == phase1_end:
        phase2_end = len(trajectory) * 3 // 4

    phases = []

    # Phase I
    if phase1_end > 0:
        phases.append(PhaseTransition(
            phase_name="routing_formation",
            start_step=int(steps[0]),
            end_step=int(steps[phase1_end]),
            alpha_start=float(alphas[0]),
            alpha_end=float(alphas[phase1_end]),
            alignment_start=float(alignments[0]),
            alignment_end=float(alignments[phase1_end]),
        ))

    # Phase II
    if phase2_end > phase1_end:
        phases.append(PhaseTransition(
            phase_name="specialization",
            start_step=int(steps[phase1_end]),
            end_step=int(steps[phase2_end]),
            alpha_start=float(alphas[phase1_end]),
            alpha_end=float(alphas[phase2_end]),
            alignment_start=float(alignments[phase1_end]),
            alignment_end=float(alignments[phase2_end]),
        ))

    # Phase III
    if phase2_end < len(trajectory) - 1:
        phases.append(PhaseTransition(
            phase_name="saturation",
            start_step=int(steps[phase2_end]),
            end_step=int(steps[-1]),
            alpha_start=float(alphas[phase2_end]),
            alpha_end=float(alphas[-1]),
            alignment_start=float(alignments[phase2_end]),
            alignment_end=float(alignments[-1]),
        ))

    return phases


# ---------------------------------------------------------------------------
# U2: KWW glass relaxation fitting for MoE
# ---------------------------------------------------------------------------

def fit_kww_moe(trajectory: list[dict], metric: str = "alpha_moe") -> dict:
    """U2: Fit KWW stretched exponential to spectral relaxation.

    φ(t) = exp[-(t/τ)^β]

    Returns dict with tau, beta, r_squared, bic_kww, bic_exp.
    """
    values = [r.get(metric, 0) for r in trajectory]
    steps = [r["step"] for r in trajectory]

    if len(values) < 5:
        return {"error": "insufficient data"}

    values = np.array(values, dtype=float)
    steps = np.array(steps, dtype=float)

    # Normalize: φ(t) = (val(t) - val_final) / (val_0 - val_final)
    v0, vf = values[0], values[-1]
    if abs(v0 - vf) < 1e-6:
        return {"error": "no relaxation observed"}

    phi = (values - vf) / (v0 - vf)
    phi = np.clip(phi, 1e-6, 1.0)
    t = steps - steps[0]
    t[0] = 1  # avoid log(0)

    def kww(t, tau, beta):
        return np.exp(-(t / tau) ** beta)

    def simple_exp(t, tau):
        return np.exp(-t / tau)

    try:
        popt_kww, _ = curve_fit(kww, t, phi, p0=[t[-1]/3, 0.7],
                                bounds=([t[1], 0.1], [t[-1]*10, 1.0]), maxfev=5000)
        pred_kww = kww(t, *popt_kww)
        ss_kww = np.sum((phi - pred_kww) ** 2)
    except Exception:
        return {"error": "KWW fit failed"}

    try:
        popt_exp, _ = curve_fit(simple_exp, t, phi, p0=[t[-1]/3],
                                bounds=([t[1]], [t[-1]*10]), maxfev=5000)
        pred_exp = simple_exp(t, *popt_exp)
        ss_exp = np.sum((phi - pred_exp) ** 2)
    except Exception:
        ss_exp = float('inf')
        popt_exp = [0]

    n = len(phi)
    ss_tot = np.sum((phi - phi.mean()) ** 2)
    r2_kww = 1 - ss_kww / ss_tot if ss_tot > 0 else 0

    bic_kww = n * np.log(ss_kww / n + 1e-15) + 2 * np.log(n)
    bic_exp = n * np.log(ss_exp / n + 1e-15) + 1 * np.log(n)

    return {
        "tau": float(popt_kww[0]),
        "beta": float(popt_kww[1]),
        "r_squared": float(r2_kww),
        "bic_kww": float(bic_kww),
        "bic_exp": float(bic_exp),
        "kww_preferred": bic_kww < bic_exp,
        "tau_exp": float(popt_exp[0]) if popt_exp[0] > 0 else None,
    }


# ---------------------------------------------------------------------------
# N8: Supercollapse analysis
# ---------------------------------------------------------------------------

def analyze_supercollapse(
    trajectories: dict[str, list[dict]],
    normalize_by: str = "active",
) -> dict:
    """N8: Check if MoE loss curves collapse when normalized.

    trajectories: {model_name: [checkpoint_records]}
    normalize_by: "total", "active", or "per_expert"

    Returns collapse quality metrics.
    """
    normalized_curves = {}

    for model_name, traj in trajectories.items():
        if not traj:
            continue

        losses = [r.get("alpha_moe", 0) for r in traj]  # use α as proxy
        steps = [r["step"] for r in traj]

        if normalize_by == "total":
            n = traj[0]["total_params"]
        elif normalize_by == "active":
            n = traj[0].get("active_params", traj[0]["total_params"])
        else:
            ne = traj[0].get("num_experts", 1)
            n = traj[0]["total_params"] / max(ne, 1)

        # Normalize step by N (compute-tokens per param)
        t_norm = np.array(steps) / n
        v_norm = np.array(losses)

        # Normalize values to [0, 1]
        if v_norm.max() > v_norm.min():
            v_norm = (v_norm - v_norm.min()) / (v_norm.max() - v_norm.min())

        normalized_curves[model_name] = (t_norm, v_norm)

    # Compute pairwise curve distances
    if len(normalized_curves) < 2:
        return {"error": "need at least 2 models"}

    names = list(normalized_curves.keys())
    distances = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            t1, v1 = normalized_curves[names[i]]
            t2, v2 = normalized_curves[names[j]]

            # Interpolate to common grid
            t_min = max(t1.min(), t2.min())
            t_max = min(t1.max(), t2.max())
            if t_min >= t_max:
                continue
            t_common = np.linspace(t_min, t_max, 100)
            v1_interp = np.interp(t_common, t1, v1)
            v2_interp = np.interp(t_common, t2, v2)

            dist = np.mean((v1_interp - v2_interp) ** 2)
            distances.append(dist)

    return {
        "normalize_by": normalize_by,
        "mean_distance": float(np.mean(distances)) if distances else None,
        "collapse_quality": 1.0 / (1.0 + float(np.mean(distances))) if distances else None,
        "n_models": len(normalized_curves),
        "n_pairs": len(distances),
    }


# ---------------------------------------------------------------------------
# Comprehensive hypothesis report
# ---------------------------------------------------------------------------

def generate_hypothesis_report(
    trajectory_path: Optional[str] = None,
    cross_model_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Generate a comprehensive hypothesis testing report.

    Args:
        trajectory_path: JSONL with multi-checkpoint measurements (e.g., OLMoE)
        cross_model_path: JSONL with cross-model final-checkpoint measurements
        output_path: where to save the report JSON
    """
    report = {"hypotheses": {}}

    # Load data
    trajectory = load_moe_trajectory(trajectory_path) if trajectory_path else []
    cross_model = load_moe_trajectory(cross_model_path) if cross_model_path else []

    # M1: SR/d convergence
    if trajectory:
        srd_result = analyze_srd_convergence(trajectory)
        if srd_result:
            report["hypotheses"]["M1_srd_convergence"] = {
                "srd_init": srd_result.srd_init,
                "srd_final": srd_result.srd_final,
                "delta_h2": srd_result.delta_h2,
                "in_dense_range": srd_result.in_dense_range,
                "convergence_step": srd_result.convergence_step,
                "verdict": "CONFIRMED" if srd_result.in_dense_range else "DIFFERS_FROM_DENSE",
            }

    # M2: α reversal
    if trajectory:
        reversals = detect_alpha_reversal(trajectory)
        report["hypotheses"]["M2_alpha_reversal"] = {
            "n_reversals": len(reversals),
            "global_reversal": any(r.layer_idx == -1 for r in reversals),
            "layer_specific_reversals": [
                {"step": r.step, "layer": r.layer_idx, "delta_alpha": r.delta_alpha}
                for r in reversals if r.layer_idx >= 0
            ][:10],
            "verdict": "EXPERT_SPECIFIC" if reversals and not any(r.layer_idx == -1 for r in reversals)
                       else "GLOBAL" if any(r.layer_idx == -1 for r in reversals)
                       else "NO_REVERSAL",
        }

    # M4: Phase transition
    if cross_model:
        pt_results = analyze_phase_transition(cross_model)
        report["hypotheses"]["M4_phase_transition"] = {
            "fits": [
                {"param_type": r.param_type, "threshold_n": r.threshold_n,
                 "r_squared": r.r_squared}
                for r in pt_results
            ],
            "best_fit": max(pt_results, key=lambda r: r.r_squared).param_type if pt_results else None,
        }

    # N1: Expert collapse
    if trajectory:
        collapse = detect_expert_collapse(trajectory)
        report["hypotheses"]["N1_expert_collapse"] = {
            "n_signals": len(collapse),
            "earliest_step": collapse[0].step if collapse else None,
            "affected_layers": list(set(c.layer_idx for c in collapse)),
        }

    # N3: Energy equipartition
    if trajectory:
        epr_results = analyze_equipartition(trajectory)
        if epr_results:
            report["hypotheses"]["N3_equipartition"] = {
                "epr_init": epr_results[0].epr_mean,
                "epr_final": epr_results[-1].epr_mean,
                "trend": epr_results[-1].epr_trend,
                "specialization_detected": epr_results[-1].epr_mean > epr_results[0].epr_mean * 1.5,
            }

    # N5: Three-phase dynamics
    if trajectory:
        phases = detect_three_phases(trajectory)
        report["hypotheses"]["N5_three_phases"] = {
            "n_phases_detected": len(phases),
            "phases": [
                {"name": p.phase_name, "start": p.start_step, "end": p.end_step,
                 "alpha_range": f"{p.alpha_start:.2f}→{p.alpha_end:.2f}"}
                for p in phases
            ],
        }

    # U2: KWW relaxation
    if trajectory:
        kww = fit_kww_moe(trajectory)
        report["hypotheses"]["U2_kww_relaxation"] = kww

    # N8: Supercollapse (needs multiple trajectories)
    # This would need trajectories from multiple models, skip if single model

    # Summary
    confirmed = sum(1 for h in report["hypotheses"].values()
                    if isinstance(h, dict) and h.get("verdict") in ("CONFIRMED", "EXPERT_SPECIFIC"))
    total = len(report["hypotheses"])
    report["summary"] = {
        "hypotheses_tested": total,
        "confirmed_or_partial": confirmed,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MoE hypothesis analysis")
    parser.add_argument("--trajectory", help="JSONL with training trajectory")
    parser.add_argument("--cross-model", help="JSONL with cross-model data")
    parser.add_argument("--output", "-o", default="results/moe_analysis/report.json")
    args = parser.parse_args()

    report = generate_hypothesis_report(
        trajectory_path=args.trajectory,
        cross_model_path=args.cross_model,
        output_path=args.output,
    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
