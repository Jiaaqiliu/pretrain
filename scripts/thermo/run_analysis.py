"""Post-hoc analysis: fit state equations, KWW relaxation, generate figures.

Runs after measurement and training experiments. CPU-only.

Usage:
    python scripts/thermo/run_analysis.py \
        --results-dir /fsx/dev/jiaqi/thermo_results \
        --experiments-dir /fsx/dev/jiaqi/thermo_experiments \
        --output-dir /fsx/dev/jiaqi/thermo_paper_figures
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Thermodynamic analysis and figure generation")
    parser.add_argument("--results-dir", required=True, help="Directory with measurement JSONL files")
    parser.add_argument("--experiments-dir", required=True, help="Directory with proxy experiment results")
    parser.add_argument("--output-dir", required=True, help="Output directory for figures and tables")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation (analysis only)")
    return parser.parse_args()


def load_measurements(results_dir: str, model_size: str) -> list[dict]:
    """Load measurement JSONL for a given model size."""
    path = Path(results_dir) / f"measurements_{model_size.lower()}.jsonl"
    records = []
    if not path.exists():
        # Try glob for sharded files
        for p in Path(results_dir).glob(f"measurements_{model_size.lower()}.rank*.jsonl"):
            with open(p) as f:
                for line in f:
                    record = json.loads(line)
                    if "error" not in record:
                        records.append(record)
    else:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if "error" not in record:
                    records.append(record)
    return sorted(records, key=lambda r: r["step"])


def load_experiment_thermo(experiments_dir: str, model_size: str, schedule: str, seed: int) -> list[dict]:
    """Load thermodynamic measurements from a proxy training experiment."""
    path = Path(experiments_dir) / f"{model_size.lower()}_{schedule}_s{seed}" / "thermo_measurements.jsonl"
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
    return sorted(records, key=lambda r: r["step"])


def records_to_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    """Convert list of record dicts to dict of arrays."""
    if not records:
        return {}
    keys = ["step", "volume", "spectral_entropy", "order_parameter",
            "internal_energy", "temperature", "pressure", "free_energy",
            "pv_over_nt", "lr"]
    result = {}
    for key in keys:
        values = [r.get(key, 0.0) for r in records]
        result[key] = np.array(values, dtype=float)
    return result


# =============================================================================
# Q1: State Equation Analysis
# =============================================================================


def analyze_state_equation(results_dir: str, output_dir: str):
    """Fit state equation across all scales."""
    from experiments.thermodynamics.analysis import (
        fit_state_equation,
        fit_k_eff,
        ideal_gas_ratio,
        identify_thermodynamic_regime,
    )

    print("\n" + "=" * 60)
    print("Q1: State Equation at Scale")
    print("=" * 60)

    model_sizes = ["190M", "1B", "7B", "13B"]
    N_values = [190e6, 1e9, 7e9, 13e9]
    k_eff_values = []
    fits = {}

    for size, N in zip(model_sizes, N_values):
        records = load_measurements(results_dir, size)
        if not records:
            print(f"  {size}: no data, skipping")
            continue

        data = records_to_arrays(records)
        P = data["pressure"]
        V = data["volume"]
        T = data["temperature"]

        # Fit state equation
        fit = fit_state_equation(P, V, int(N), T)
        fits[size] = fit

        # Identify stable phase (assume first 60-80% of training)
        n = len(data["step"])
        stable_mask = np.zeros(n, dtype=bool)
        stable_mask[int(0.02 * n):int(0.8 * n)] = True

        k_eff_stable = np.nanmedian(data["pv_over_nt"][stable_mask])
        k_eff_values.append(k_eff_stable)

        # Identify regimes
        regimes = identify_thermodynamic_regime(T, data["pv_over_nt"])
        n_ideal = (regimes == "ideal_gas").sum()
        n_liquid = (regimes == "liquid").sum()
        n_glass = (regimes == "glass").sum()

        print(f"  OLMo-{size}: k_eff={k_eff_stable:.4f}, R²={fit.r_squared:.4f}, "
              f"regimes: ideal={n_ideal}, liquid={n_liquid}, glass={n_glass}")

    # Fit k_eff(N) = k₀ + α·N^(-1/3) across scales
    if len(k_eff_values) >= 2:
        valid_N = np.array(N_values[:len(k_eff_values)])
        valid_k = np.array(k_eff_values)
        k0, alpha = fit_k_eff(valid_k, valid_N)
        print(f"\n  Scaling law: k_eff(N) = {k0:.4f} + {alpha:.4f} · N^(-1/3)")
    else:
        print("\n  Insufficient data for scaling law fit")

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "q1_state_equation.json", "w") as f:
        json.dump({
            "fits": {size: {"k_eff": fit.k_eff, "alpha": fit.alpha, "gamma": fit.gamma,
                            "r_squared": fit.r_squared, "num_points": fit.num_points}
                     for size, fit in fits.items()},
            "scaling_law": {"k0": k0 if len(k_eff_values) >= 2 else None,
                            "alpha": alpha if len(k_eff_values) >= 2 else None},
        }, f, indent=2)

    return fits


# =============================================================================
# Q2: WSD vs Cosine
# =============================================================================


def analyze_wsd_vs_cosine(results_dir: str, output_dir: str):
    """Compare entropy production between WSD and cosine schedules."""
    from experiments.thermodynamics.measures import (
        entropy_production_rate,
        cumulative_entropy_production,
        thermodynamic_efficiency,
    )

    print("\n" + "=" * 60)
    print("Q2: WSD vs Cosine Entropy Comparison")
    print("=" * 60)

    # This requires either:
    # 1. Existing OLMo checkpoints trained with both WSD and cosine
    # 2. Our proxy experiment results
    # For now, analyze measurements if available

    model_sizes = ["190M", "1B", "7B"]
    comparisons = {}

    for size in model_sizes:
        records = load_measurements(results_dir, size)
        if not records:
            continue

        data = records_to_arrays(records)

        # Compute entropy production rate
        sigma = entropy_production_rate(
            data["free_energy"], data["temperature"], data["step"]
        )
        cum_s = cumulative_entropy_production(sigma, data["step"])

        total_entropy = cum_s[-1] if len(cum_s) > 0 else 0.0
        delta_f = abs(data["free_energy"][-1] - data["free_energy"][0])
        mean_t = np.mean(data["temperature"][data["temperature"] > 0])
        efficiency = thermodynamic_efficiency(delta_f, total_entropy, mean_t)

        comparisons[size] = {
            "total_entropy_production": total_entropy,
            "delta_free_energy": delta_f,
            "efficiency": efficiency,
            "mean_temperature": mean_t,
        }

        print(f"  OLMo-{size}: ΔS_tot={total_entropy:.4f}, ΔF={delta_f:.4f}, "
              f"η_thermo={efficiency:.4f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "q2_entropy_comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2)


# =============================================================================
# Q3: KWW Relaxation
# =============================================================================


def analyze_kww(results_dir: str, output_dir: str):
    """Fit KWW stretched exponential to mid-training spectral entropy."""
    from experiments.thermodynamics.analysis import fit_kww

    print("\n" + "=" * 60)
    print("Q3: KWW Glass Relaxation Fitting")
    print("=" * 60)

    model_sizes = ["190M", "1B", "7B"]
    kww_results = {}

    for size in model_sizes:
        records = load_measurements(results_dir, size)
        if not records:
            continue

        data = records_to_arrays(records)

        # Detect mid-training onset (look for a drop in spectral entropy > 2%)
        s = data["spectral_entropy"]
        steps = data["step"]

        onset_step = None
        for i in range(1, len(s)):
            if s[i] < s[i - 1] * 0.98:  # 2% drop
                onset_step = int(steps[i])
                break

        if onset_step is None:
            # Default to 60% of training
            onset_step = int(steps[int(0.6 * len(steps))])
            print(f"  OLMo-{size}: no clear onset, using default at step {onset_step}")

        kww = fit_kww(s, steps, onset_step)

        kww_results[size] = {
            "tau": kww.tau,
            "beta": kww.beta,
            "r_squared": kww.r_squared,
            "s_0": kww.s_0,
            "s_inf": kww.s_inf,
            "tau_ci": list(kww.tau_ci),
            "beta_ci": list(kww.beta_ci),
            "bic_kww": kww.bic_kww,
            "bic_exponential": kww.bic_exponential,
            "bic_powerlaw": kww.bic_powerlaw,
            "optimal_duration_3tau": kww.optimal_duration_3tau,
            "onset_step": onset_step,
        }

        print(f"  OLMo-{size}: τ={kww.tau:.1f}, β={kww.beta:.3f}, R²={kww.r_squared:.4f}")
        print(f"    BIC: KWW={kww.bic_kww:.1f}, Exp={kww.bic_exponential:.1f}, "
              f"PL={kww.bic_powerlaw:.1f}")
        print(f"    Optimal mid-training duration: 3τ = {kww.optimal_duration_3tau:.0f} steps")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "q3_kww_fitting.json", "w") as f:
        json.dump(kww_results, f, indent=2)


# =============================================================================
# Q5: Schedule Comparison
# =============================================================================


def analyze_schedule_comparison(experiments_dir: str, output_dir: str):
    """Statistical comparison of training schedules from proxy experiments."""
    from experiments.thermodynamics.analysis import compare_schedules

    print("\n" + "=" * 60)
    print("Q5: Schedule Comparison")
    print("=" * 60)

    model_sizes = ["190M", "1B"]
    schedules = ["cosine", "wsd_linear", "wsd_exponential", "gaussian"]
    seeds = [42, 123, 456]

    all_results = {}

    for size in model_sizes:
        results = {}
        for schedule in schedules:
            losses = []
            for seed in seeds:
                records = load_experiment_thermo(experiments_dir, size, schedule, seed)
                if records:
                    final_loss = records[-1].get("loss", None)
                    if final_loss is not None:
                        losses.append(final_loss)

            if losses:
                results[schedule] = losses
                mean_loss = np.mean(losses)
                std_loss = np.std(losses)
                print(f"  OLMo-{size} / {schedule}: loss={mean_loss:.4f} ± {std_loss:.4f} ({len(losses)} seeds)")

        if len(results) >= 2:
            comparisons = compare_schedules(results, baseline="wsd_linear")
            for comp in comparisons:
                print(f"    {comp.schedule} vs WSD-Linear: Δ={comp.delta_pct:+.2f}%, "
                      f"p={comp.p_value:.4f} {'*' if comp.significant else ''}")
            all_results[size] = {
                "raw": {k: v for k, v in results.items()},
                "comparisons": [{
                    "schedule": c.schedule,
                    "delta_pct": c.delta_pct,
                    "p_value": c.p_value,
                    "significant": c.significant,
                } for c in comparisons],
            }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "q5_schedule_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)


# =============================================================================
# Figure Generation
# =============================================================================


def generate_figures(results_dir: str, experiments_dir: str, output_dir: str):
    """Generate all paper figures."""
    from experiments.thermodynamics import viz

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Fig 0: LR schedules comparison
    viz.plot_lr_schedules(
        total_steps=10000, peak_lr=6e-4,
        save_path=str(out / "fig0_lr_schedules.pdf"),
    )
    print("Generated: fig0_lr_schedules.pdf")

    # Per-scale figures
    for size in ["190M", "1B", "7B", "13B"]:
        records = load_measurements(results_dir, size)
        if not records:
            continue

        data = records_to_arrays(records)

        # Phase diagram
        viz.plot_phase_diagram(
            temperature=data["temperature"],
            spectral_entropy=data["spectral_entropy"],
            steps=data["step"],
            schedule_name="WSD",
            save_path=str(out / f"fig_phase_diagram_{size}.pdf"),
        )
        print(f"Generated: fig_phase_diagram_{size}.pdf")

        # State equation convergence
        viz.plot_state_equation_convergence(
            steps=data["step"],
            pv_over_nt=data["pv_over_nt"],
            model_size=size,
            save_path=str(out / f"fig_state_eq_{size}.pdf"),
        )
        print(f"Generated: fig_state_eq_{size}.pdf")

    # Schedule comparison for proxy experiments
    for size in ["190M", "1B"]:
        trajectories = {}
        for schedule in ["cosine", "wsd_linear", "wsd_exponential", "gaussian"]:
            records = load_experiment_thermo(experiments_dir, size, schedule, 42)
            if records:
                data = records_to_arrays(records)
                trajectories[schedule] = (data["temperature"], data["spectral_entropy"])

        if trajectories:
            viz.plot_ts_trajectories(
                trajectories=trajectories,
                model_size=size,
                save_path=str(out / f"fig_ts_trajectories_{size}.pdf"),
            )
            print(f"Generated: fig_ts_trajectories_{size}.pdf")


def main():
    args = parse_args()

    # Run all analyses
    analyze_state_equation(args.results_dir, args.output_dir)
    analyze_wsd_vs_cosine(args.results_dir, args.output_dir)
    analyze_kww(args.results_dir, args.output_dir)
    analyze_schedule_comparison(args.experiments_dir, args.output_dir)

    # Generate figures
    if not args.skip_figures:
        generate_figures(args.results_dir, args.experiments_dir, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
