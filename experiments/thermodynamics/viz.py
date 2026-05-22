"""Visualization for thermodynamics of pretraining paper figures.

Generates publication-quality figures for NeurIPS 2026:
  - Phase diagrams (T-S plane)
  - State equation convergence plots
  - KWW relaxation fits
  - Correlation heatmaps
  - Schedule comparison bar charts
"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

COLORS = {
    "wsd_linear": "#2196F3",
    "wsd_exponential": "#FF9800",
    "cosine": "#4CAF50",
    "gaussian": "#E91E63",
    "ideal_gas": "#66BB6A",
    "liquid": "#42A5F5",
    "glass": "#EF5350",
}

NEURIPS_RC = {
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (5.5, 3.5),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def _setup():
    if not HAS_MPL:
        raise ImportError("matplotlib required: pip install matplotlib")
    plt.rcParams.update(NEURIPS_RC)


def plot_phase_diagram(
    temperature: np.ndarray,
    spectral_entropy: np.ndarray,
    steps: np.ndarray,
    schedule_name: str = "WSD",
    regimes: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    ax=None,
):
    """Plot T-S phase diagram (Figure 1 in paper).

    Training trajectory in temperature-entropy plane, colored by regime.
    """
    _setup()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots()

    if regimes is not None:
        for regime, color in [("ideal_gas", COLORS["ideal_gas"]),
                              ("liquid", COLORS["liquid"]),
                              ("glass", COLORS["glass"])]:
            mask = regimes == regime
            if mask.any():
                ax.scatter(
                    temperature[mask], spectral_entropy[mask],
                    c=color, s=8, alpha=0.6, label=regime.replace("_", " ").title(),
                    edgecolors="none",
                )
    else:
        sc = ax.scatter(
            temperature, spectral_entropy,
            c=steps, s=8, alpha=0.7, cmap="viridis",
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="Training step")

    # Arrow showing direction of time
    n = len(temperature)
    mid = n // 2
    if mid > 0 and mid < n - 1:
        ax.annotate(
            "", xy=(temperature[mid + 1], spectral_entropy[mid + 1]),
            xytext=(temperature[mid], spectral_entropy[mid]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

    ax.set_xlabel(r"Temperature $T$")
    ax.set_ylabel(r"Spectral Entropy $S$")
    ax.set_title(f"Phase Diagram ({schedule_name})")
    if regimes is not None:
        ax.legend(framealpha=0.8)

    if save_path and own_fig:
        fig.savefig(save_path)
        plt.close(fig)
    return ax


def plot_state_equation_convergence(
    steps: np.ndarray,
    pv_over_nt: np.ndarray,
    k_eff_fitted: Optional[float] = None,
    schedule_boundaries: Optional[dict[str, int]] = None,
    model_size: str = "",
    save_path: Optional[str] = None,
):
    """Plot P·V/(N·T) convergence over training (Q1, Figure 2)."""
    _setup()
    fig, ax = plt.subplots()

    ax.plot(steps, pv_over_nt, color=COLORS["wsd_linear"], alpha=0.8, label=r"$PV/(NT)$")

    if k_eff_fitted is not None:
        ax.axhline(k_eff_fitted, color="gray", linestyle="--", alpha=0.5,
                    label=rf"$k_{{\mathrm{{eff}}}} = {k_eff_fitted:.3f}$")

    if schedule_boundaries:
        for name, step in schedule_boundaries.items():
            ax.axvline(step, color="red", linestyle=":", alpha=0.4, label=name)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$PV / (NT)$")
    ax.set_title(f"State Equation Convergence — OLMo-{model_size}")
    ax.legend(framealpha=0.8)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_entropy_comparison(
    steps_wsd: np.ndarray,
    cum_entropy_wsd: np.ndarray,
    steps_cos: np.ndarray,
    cum_entropy_cos: np.ndarray,
    model_size: str = "",
    save_path: Optional[str] = None,
):
    """Plot cumulative entropy production: WSD vs Cosine (Q2, Figure 3)."""
    _setup()
    fig, ax = plt.subplots()

    ax.plot(steps_wsd, cum_entropy_wsd, color=COLORS["wsd_linear"],
            label="WSD-Linear", linewidth=2)
    ax.plot(steps_cos, cum_entropy_cos, color=COLORS["cosine"],
            label="Cosine", linewidth=2)

    # Shade the gap
    steps_common = np.intersect1d(steps_wsd, steps_cos)
    if len(steps_common) > 0:
        idx_wsd = np.searchsorted(steps_wsd, steps_common)
        idx_cos = np.searchsorted(steps_cos, steps_common)
        ax.fill_between(
            steps_common,
            cum_entropy_wsd[idx_wsd],
            cum_entropy_cos[idx_cos],
            alpha=0.15, color="gray",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"Cumulative Entropy Production $\Delta S_{\mathrm{tot}}$")
    ax.set_title(f"Entropy Production: WSD vs Cosine — OLMo-{model_size}")
    ax.legend(framealpha=0.8)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_ts_trajectories(
    trajectories: dict[str, tuple[np.ndarray, np.ndarray]],
    model_size: str = "",
    save_path: Optional[str] = None,
):
    """Plot T-S phase space trajectories for multiple schedules (Q2, Figure 4)."""
    _setup()
    fig, ax = plt.subplots()

    for schedule_name, (T, S) in trajectories.items():
        color = COLORS.get(schedule_name, "gray")
        ax.plot(T, S, color=color, label=schedule_name.replace("_", " ").title(),
                linewidth=2, alpha=0.8)

        # Start/end markers
        ax.scatter([T[0]], [S[0]], color=color, marker="o", s=40, zorder=5)
        ax.scatter([T[-1]], [S[-1]], color=color, marker="s", s=40, zorder=5)

    ax.set_xlabel(r"Temperature $T$ (effective)")
    ax.set_ylabel(r"Spectral Entropy $S$")
    ax.set_title(f"Phase-Space Trajectories — OLMo-{model_size}")
    ax.legend(framealpha=0.8)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_kww_fit(
    steps: np.ndarray,
    phi_measured: np.ndarray,
    phi_kww: np.ndarray,
    phi_exp: Optional[np.ndarray] = None,
    tau: float = 0.0,
    beta: float = 0.0,
    r_squared: float = 0.0,
    model_size: str = "",
    save_path: Optional[str] = None,
):
    """Plot KWW relaxation fit (Q3, Figure 5)."""
    _setup()
    fig, ax = plt.subplots()

    ax.scatter(steps, phi_measured, s=10, alpha=0.4, color="gray", label="Measured")
    ax.plot(steps, phi_kww, color=COLORS["gaussian"], linewidth=2,
            label=rf"KWW ($\tau$={tau:.0f}, $\beta$={beta:.2f})")
    if phi_exp is not None:
        ax.plot(steps, phi_exp, color=COLORS["cosine"], linewidth=1.5,
                linestyle="--", label="Simple Exponential")

    ax.set_xlabel("Steps since mid-training onset")
    ax.set_ylabel(r"Normalized relaxation $\phi(t)$")
    ax.set_title(
        f"KWW Relaxation Fit — OLMo-{model_size} ($R^2$={r_squared:.3f})"
    )
    ax.legend(framealpha=0.8)
    ax.set_ylim(-0.05, 1.05)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_monitoring_correlation(
    correlations: list,
    model_sizes: list[str],
    save_path: Optional[str] = None,
):
    """Plot correlation heatmap: metrics vs downstream benchmarks (Q4, Figure 6)."""
    _setup()

    metrics = [c.metric_name for c in correlations]
    n_metrics = len(metrics)
    n_sizes = len(model_sizes)

    data = np.zeros((n_metrics, n_sizes))
    for i, corr in enumerate(correlations):
        for j, size in enumerate(model_sizes):
            data[i, j] = corr.per_benchmark.get(size, corr.spearman_r)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels(model_sizes)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels([m.replace("_", " ").title() for m in metrics])

    # Annotate cells
    for i in range(n_metrics):
        for j in range(n_sizes):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label="Spearman r")
    ax.set_title("Metric-Benchmark Correlation")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_schedule_comparison(
    comparisons: list,
    model_size: str = "",
    save_path: Optional[str] = None,
):
    """Plot schedule comparison bar chart (Q5, Figure 7)."""
    _setup()
    fig, ax = plt.subplots()

    schedules = ["cosine", "wsd_linear", "wsd_exponential", "gaussian"]
    display_names = ["Cosine", "WSD-Linear", "WSD-Exp", "Gaussian (ours)"]
    colors = [COLORS.get(s, "gray") for s in schedules]

    # Gather data
    deltas = [0.0] * len(schedules)
    errors = [0.0] * len(schedules)
    for comp in comparisons:
        if comp.schedule in schedules:
            idx = schedules.index(comp.schedule)
            deltas[idx] = comp.delta_pct
            errors[idx] = comp.std_error / max(comp.baseline_mean_loss, 1e-10) * 100
    # Baseline is 0
    baseline_idx = schedules.index("wsd_linear")
    deltas[baseline_idx] = 0.0

    bars = ax.bar(range(len(schedules)), deltas, color=colors, alpha=0.8,
                  yerr=errors, capsize=4, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(schedules)))
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.set_ylabel(r"$\Delta$ Loss vs WSD-Linear (\%)")
    ax.set_title(f"Schedule Comparison — OLMo-{model_size}")

    # Annotate bars
    for bar, delta in zip(bars, deltas):
        if delta != 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1 * np.sign(bar.get_height()),
                f"{delta:+.1f}%", ha="center", va="bottom" if delta > 0 else "top",
                fontsize=7,
            )

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_lr_schedules(
    total_steps: int = 10000,
    peak_lr: float = 6e-4,
    save_path: Optional[str] = None,
):
    """Plot all four LR schedules for visual comparison."""
    _setup()
    from experiments.thermodynamics.schedules import SCHEDULE_REGISTRY

    fig, ax = plt.subplots()
    steps = np.arange(total_steps)

    for name, schedule_fn in SCHEDULE_REGISTRY.items():
        lrs = [schedule_fn(s, total_steps, peak_lr) for s in steps]
        ax.plot(steps, lrs, color=COLORS.get(name, "gray"),
                label=name.replace("_", " ").title(), linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedules")
    ax.legend(framealpha=0.8)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def generate_all_figures(
    results_dir: str,
    output_dir: str,
):
    """Generate all paper figures from saved measurement results.

    Expects results_dir to contain:
      - measurements_{size}.jsonl for each model size
      - training_{schedule}_{size}_{seed}.jsonl for proxy experiments
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Figure 0: LR schedule comparison
    plot_lr_schedules(save_path=str(output / "fig0_lr_schedules.pdf"))

    print(f"Figures saved to {output_dir}")
