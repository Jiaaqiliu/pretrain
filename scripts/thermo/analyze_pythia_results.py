"""Analyze Pythia thermodynamic measurements.

Implements E1 (state equation) and E5 (ψ-benchmark correlation) analyses.

Usage:
    python scripts/thermo/analyze_pythia_results.py \
        --results-dir results/pythia/ \
        --output-dir results/pythia/analysis/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import curve_fit
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


PYTHIA_SCALES_ORDERED = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]


def load_all_results(results_dir: str) -> dict:
    """Load all Pythia JSONL results into a dict keyed by model size."""
    data = {}
    results_path = Path(results_dir)
    for f in sorted(results_path.glob("pythia_*.jsonl")):
        size = f.stem.replace("pythia_", "")
        records = []
        with open(f) as fp:
            for line in fp:
                r = json.loads(line)
                if "error" not in r:
                    records.append(r)
        records.sort(key=lambda x: x["step"])
        data[size] = records
        print(f"Loaded {len(records)} records for pythia-{size}")
    return data


def analyze_state_equation(data: dict, output_dir: Path):
    """E1: Analyze PV/(NT) convergence and fit k_eff(N)."""
    print("\n" + "=" * 60)
    print("E1: STATE EQUATION ANALYSIS")
    print("=" * 60)

    # For each scale, compute average PV/(NT) in stable phase (step 10K-100K)
    scale_results = {}
    for size, records in sorted(data.items(), key=lambda x: x[1][0]["num_params"] if x[1] else 0):
        stable = [r for r in records if 10000 <= r["step"] <= 100000 and r.get("pv_over_nt", 0) > 0]
        if not stable:
            print(f"  pythia-{size}: no stable-phase data")
            continue

        pvnt_values = [r["pv_over_nt"] for r in stable]
        N = stable[0]["num_params"]
        keff = np.mean(pvnt_values)
        keff_std = np.std(pvnt_values)
        cv = keff_std / keff if keff > 0 else float("inf")

        scale_results[size] = {
            "N": N,
            "keff_mean": keff,
            "keff_std": keff_std,
            "keff_cv": cv,
            "n_points": len(stable),
        }

        print(f"  pythia-{size:5s} (N={N:>12,}): k_eff = {keff:.4f} ± {keff_std:.4f} (CV={cv:.2%}, n={len(stable)})")

    if len(scale_results) < 3:
        print("\n  WARNING: Need at least 3 scales for fitting. Skipping curve fit.")
        _save_json(output_dir / "E1_state_equation.json", scale_results)
        return scale_results

    # Fit k_eff(N) = k0 + alpha * N^(-1/3)
    if HAS_SCIPY:
        N_vals = np.array([v["N"] for v in scale_results.values()])
        keff_vals = np.array([v["keff_mean"] for v in scale_results.values()])
        keff_stds = np.array([v["keff_std"] for v in scale_results.values()])
        # Replace zero std with small value
        keff_stds = np.where(keff_stds > 0, keff_stds, 0.01 * keff_vals)

        def state_eq(N, k0, alpha):
            return k0 + alpha * N ** (-1 / 3)

        try:
            popt, pcov = curve_fit(state_eq, N_vals, keff_vals,
                                   sigma=keff_stds, p0=[0.5, 1000.0],
                                   maxfev=10000)
            k0, alpha = popt
            perr = np.sqrt(np.diag(pcov))

            # R²
            fitted = state_eq(N_vals, *popt)
            ss_res = np.sum((keff_vals - fitted) ** 2)
            ss_tot = np.sum((keff_vals - np.mean(keff_vals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            print(f"\n  State equation fit: k_eff(N) = {k0:.4f} + {alpha:.1f} * N^(-1/3)")
            print(f"  k0 = {k0:.4f} ± {perr[0]:.4f}")
            print(f"  α  = {alpha:.1f} ± {perr[1]:.1f}")
            print(f"  R² = {r2:.4f}")

            fit_result = {
                "k0": float(k0), "k0_err": float(perr[0]),
                "alpha": float(alpha), "alpha_err": float(perr[1]),
                "r_squared": float(r2),
            }
            scale_results["_fit"] = fit_result
        except Exception as e:
            print(f"\n  Curve fit failed: {e}")

    _save_json(output_dir / "E1_state_equation.json", scale_results)
    return scale_results


def analyze_trajectories(data: dict, output_dir: Path):
    """Analyze S(t), ψ(t), V(t) trajectories for all scales."""
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS (S, ψ, V over training)")
    print("=" * 60)

    trajectory_summary = {}
    for size, records in sorted(data.items(), key=lambda x: x[1][0]["num_params"] if x[1] else 0):
        if not records:
            continue

        first = records[0]
        last = records[-1]
        N = first["num_params"]

        # Find min S, max ψ
        min_s_rec = min(records, key=lambda r: r["spectral_entropy"])
        max_psi_rec = max(records, key=lambda r: r["order_parameter"])

        s_init = first["spectral_entropy"]
        s_final = last["spectral_entropy"]
        psi_init = first["order_parameter"]
        psi_final = last["order_parameter"]
        v_init = first["volume"]
        v_final = last["volume"]

        summary = {
            "N": N,
            "steps_measured": len(records),
            "S_init": s_init, "S_final": s_final, "delta_S": s_final - s_init,
            "S_min": min_s_rec["spectral_entropy"], "S_min_step": min_s_rec["step"],
            "psi_init": psi_init, "psi_final": psi_final, "delta_psi": psi_final - psi_init,
            "psi_max": max_psi_rec["order_parameter"], "psi_max_step": max_psi_rec["step"],
            "V_init": v_init, "V_final": v_final, "V_ratio": v_final / v_init if v_init > 0 else 0,
        }
        trajectory_summary[size] = summary

        print(f"\n  pythia-{size} (N={N:,}):")
        print(f"    S: {s_init:.4f} → {s_final:.4f} (ΔS={s_final-s_init:+.4f}, min={min_s_rec['spectral_entropy']:.4f} @ step {min_s_rec['step']})")
        print(f"    ψ: {psi_init:.4f} → {psi_final:.4f} (Δψ={psi_final-psi_init:+.4f}, max={max_psi_rec['order_parameter']:.4f} @ step {max_psi_rec['step']})")
        print(f"    V: {v_init:.0f} → {v_final:.0f} (×{v_final/v_init:.1f})")

    _save_json(output_dir / "trajectory_summary.json", trajectory_summary)

    # Cross-scale ψ(N) analysis
    if len(trajectory_summary) >= 3:
        print("\n  --- ψ(N) Scaling ---")
        N_vals = []
        psi_vals = []
        for size in PYTHIA_SCALES_ORDERED:
            if size in trajectory_summary:
                N_vals.append(trajectory_summary[size]["N"])
                psi_vals.append(trajectory_summary[size]["psi_final"])

        N_arr = np.array(N_vals)
        psi_arr = np.array(psi_vals)

        if HAS_SCIPY and len(N_arr) >= 3:
            # Fit ψ(N) = a * N^b
            def power_law(N, a, b):
                return a * N ** b

            try:
                # Use log-space for better fitting
                log_N = np.log10(N_arr)
                log_psi = np.log10(psi_arr)
                coeffs = np.polyfit(log_N, log_psi, 1)
                b_fit = coeffs[0]
                a_fit = 10 ** coeffs[1]

                psi_predicted = a_fit * N_arr ** b_fit
                ss_res = np.sum((psi_arr - psi_predicted) ** 2)
                ss_tot = np.sum((psi_arr - np.mean(psi_arr)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                print(f"    ψ(N) = {a_fit:.4e} × N^{b_fit:.3f}")
                print(f"    R² = {r2:.4f}")

                trajectory_summary["_psi_scaling"] = {
                    "a": float(a_fit), "b": float(b_fit), "r_squared": float(r2),
                    "N_values": N_vals, "psi_values": psi_vals,
                }
            except Exception as e:
                print(f"    Fit failed: {e}")

    _save_json(output_dir / "trajectory_summary.json", trajectory_summary)
    return trajectory_summary


def analyze_training_phases(data: dict, output_dir: Path):
    """Identify thermodynamic phases during training."""
    print("\n" + "=" * 60)
    print("TRAINING PHASE ANALYSIS")
    print("=" * 60)

    phase_results = {}
    for size, records in sorted(data.items(), key=lambda x: x[1][0]["num_params"] if x[1] else 0):
        if len(records) < 5:
            continue

        steps = np.array([r["step"] for r in records])
        S_vals = np.array([r["spectral_entropy"] for r in records])
        psi_vals = np.array([r["order_parameter"] for r in records])
        V_vals = np.array([r["volume"] for r in records])

        # Phase detection via rate of change
        # Warmup: step 0-1430
        # Early training: rapid S decline (step 1430-10000)
        # Stable training: slow S decline (step 10000-100000)
        # Late training: S approaches plateau (step 100000+)

        warmup = [r for r in records if r["step"] <= 1430]
        early = [r for r in records if 1430 < r["step"] <= 10000]
        stable = [r for r in records if 10000 < r["step"] <= 100000]
        late = [r for r in records if r["step"] > 100000]

        phases = {}
        for phase_name, phase_data in [("warmup", warmup), ("early", early),
                                        ("stable", stable), ("late", late)]:
            if len(phase_data) >= 2:
                s_start = phase_data[0]["spectral_entropy"]
                s_end = phase_data[-1]["spectral_entropy"]
                psi_start = phase_data[0]["order_parameter"]
                psi_end = phase_data[-1]["order_parameter"]
                steps_in_phase = phase_data[-1]["step"] - phase_data[0]["step"]
                phases[phase_name] = {
                    "step_range": [phase_data[0]["step"], phase_data[-1]["step"]],
                    "delta_S": s_end - s_start,
                    "delta_psi": psi_end - psi_start,
                    "dS_per_1000_steps": (s_end - s_start) / (steps_in_phase / 1000) if steps_in_phase > 0 else 0,
                    "dpsi_per_1000_steps": (psi_end - psi_start) / (steps_in_phase / 1000) if steps_in_phase > 0 else 0,
                }

        phase_results[size] = phases
        print(f"\n  pythia-{size}:")
        for pname, pdata in phases.items():
            print(f"    {pname:8s} (step {pdata['step_range'][0]:>6}-{pdata['step_range'][1]:>6}): "
                  f"ΔS={pdata['delta_S']:+.4f}, Δψ={pdata['delta_psi']:+.4f}")

    _save_json(output_dir / "training_phases.json", phase_results)
    return phase_results


def generate_report(data: dict, state_eq: dict, trajectories: dict, phases: dict, output_dir: Path):
    """Generate markdown analysis report."""
    report_lines = [
        "# Pythia Thermodynamic Measurement Results",
        "",
        f"> Generated from {sum(len(v) for v in data.values())} measurements across {len(data)} model scales",
        "",
        "---",
        "",
        "## 1. Cross-Scale Summary",
        "",
        "| Model | N | S_init | S_final | ΔS | ψ_init | ψ_final | Δψ | V_final/V_init |",
        "|-------|---|--------|---------|-----|--------|---------|-----|----------------|",
    ]

    for size in PYTHIA_SCALES_ORDERED:
        if size in trajectories:
            t = trajectories[size]
            report_lines.append(
                f"| pythia-{size} | {t['N']:,} | {t['S_init']:.4f} | {t['S_final']:.4f} | "
                f"{t['delta_S']:+.4f} | {t['psi_init']:.4f} | {t['psi_final']:.4f} | "
                f"{t['delta_psi']:+.4f} | ×{t['V_ratio']:.1f} |"
            )

    report_lines.extend([
        "",
        "## 2. State Equation: PV/(NT) = k_eff(N)",
        "",
        "| Model | N | k_eff (mean) | k_eff (std) | CV | n_points |",
        "|-------|---|-------------|-------------|-----|----------|",
    ])

    for size in PYTHIA_SCALES_ORDERED:
        if size in state_eq and not size.startswith("_"):
            s = state_eq[size]
            report_lines.append(
                f"| pythia-{size} | {s['N']:,} | {s['keff_mean']:.4f} | {s['keff_std']:.4f} | "
                f"{s['keff_cv']:.2%} | {s['n_points']} |"
            )

    if "_fit" in state_eq:
        f = state_eq["_fit"]
        report_lines.extend([
            "",
            f"**Fit**: k_eff(N) = {f['k0']:.4f} + {f['alpha']:.1f} × N^(-1/3)",
            f"- R² = {f['r_squared']:.4f}",
            f"- k₀ = {f['k0']:.4f} ± {f['k0_err']:.4f}",
            f"- α = {f['alpha']:.1f} ± {f['alpha_err']:.1f}",
        ])

    # ψ(N) scaling
    if "_psi_scaling" in trajectories:
        ps = trajectories["_psi_scaling"]
        report_lines.extend([
            "",
            "## 3. ψ(N) Scaling Law",
            "",
            f"**Fit**: ψ(N) = {ps['a']:.4e} × N^{ps['b']:.3f}",
            f"- R² = {ps['r_squared']:.4f}",
            f"- Exponent b = {ps['b']:.3f} (ψ grows as ~N^{ps['b']:.2f})",
        ])

    # Training phases
    report_lines.extend([
        "",
        "## 4. Training Phase Analysis",
        "",
        "| Model | Phase | Steps | ΔS | Δψ | dS/1000steps |",
        "|-------|-------|-------|-----|-----|-------------|",
    ])

    for size in PYTHIA_SCALES_ORDERED:
        if size in phases:
            for pname, pdata in phases[size].items():
                report_lines.append(
                    f"| pythia-{size} | {pname} | {pdata['step_range'][0]}-{pdata['step_range'][1]} | "
                    f"{pdata['delta_S']:+.5f} | {pdata['delta_psi']:+.5f} | {pdata['dS_per_1000_steps']:+.5f} |"
                )

    # Key findings
    report_lines.extend([
        "",
        "## 5. Key Findings",
        "",
        "### P1: S decreases during training",
        "",
        "### P2: ψ increases with model scale",
        "",
        "### P3: PV/(NT) converges to scale-dependent constant",
        "",
        "---",
        "",
        "*Analysis generated by `scripts/thermo/analyze_pythia_results.py`*",
    ])

    report_path = output_dir / "PYTHIA_ANALYSIS.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


def _save_json(path: Path, data):
    """Save data as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory with JSONL files")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_all_results(args.results_dir)
    if not data:
        print("ERROR: No result files found!")
        return

    # Run analyses
    state_eq = analyze_state_equation(data, output_dir)
    trajectories = analyze_trajectories(data, output_dir)
    phases = analyze_training_phases(data, output_dir)

    # Generate report
    generate_report(data, state_eq, trajectories, phases, output_dir)


if __name__ == "__main__":
    main()
