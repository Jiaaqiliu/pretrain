"""Generate publication-quality figures for NeurIPS 2026 paper.

Style: clean, minimal, high-contrast. Suitable for top-tier AI venues.
Uses seaborn for color palettes, matplotlib for fine control.

Run: python scripts/figures/plot_all.py
Output: paper/figures/*.pdf
"""

import json
import re
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# =============================================================================
# Global Style Configuration (NeurIPS-compatible)
# =============================================================================
sns.set_context("paper", font_scale=1.0)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.linewidth': 1.2,
    'grid.linewidth': 0.25,
    'grid.alpha': 0.35,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Color palettes
PALETTE_MAIN = sns.color_palette("Set2", 8)
PALETTE_SEQ = sns.color_palette("viridis", 6)
PALETTE_DIV = sns.color_palette("RdYlBu_r", 5)
ACCENT = '#E63946'
GRAY = '#6c757d'

RESULTS_DIR = Path("results")
FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

COL_WIDTH = 3.25  # single column (inches) for NeurIPS
TEXT_WIDTH = 6.75  # full text width


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r and "alpha_mean" in r:
                records.append(r)
    records.sort(key=lambda r: r["step"])
    return records


def load_log(path):
    records = []
    with open(path) as f:
        for line in f:
            m = re.search(r"step (\d+)/\d+: loss=([\d.]+), lr=([\d.e+-]+)", line)
            if m:
                records.append({"step": int(m.group(1)), "loss": float(m.group(2)), "lr": float(m.group(3))})
            m2 = re.search(r"\[SPECTRAL\] step (\d+): .=([\d.]+), SR/d=([\d.]+)", line)
            if m2:
                step = int(m2.group(1))
                for r in records:
                    if r["step"] == step:
                        r["alpha"] = float(m2.group(2))
                        r["sr_d"] = float(m2.group(3))
                        break
                else:
                    records.append({"step": step, "alpha": float(m2.group(2)), "sr_d": float(m2.group(3))})
    return records


# =============================================================================
# Figure 1: SR/d convergence across scales (single column)
# =============================================================================
def fig1_srd_convergence():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))

    pythia_models = [
        ("pythia_70m", "70M", 512),
        ("pythia_160m", "160M", 768),
        ("pythia_410m", "410M", 1024),
        ("pythia_1b", "1B", 2048),
        ("pythia_2.8b", "2.8B", 2560),
        ("pythia_6.9b", "6.9B", 4096),
    ]

    colors = sns.color_palette("flare", len(pythia_models))

    for i, (fname, label, d) in enumerate(pythia_models):
        path = RESULTS_DIR / "pythia_v2" / f"{fname}.jsonl"
        if path.exists():
            records = load_jsonl(path)
            steps = np.array([r["step"] for r in records]) / 1000
            sr_d = [r["stable_rank_mean"] / d for r in records]
            ax.plot(steps, sr_d, color=colors[i], label=f"{label}", linewidth=1.3, alpha=0.9)

    ax.axhline(y=0.055, color=ACCENT, linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)
    ax.text(105, 0.065, r"SR/$d$ $\approx$ 0.055", fontsize=7, color=ACCENT)

    ax.set_xlabel("Training Step (×10³)")
    ax.set_ylabel(r"SR/$d$")
    ax.legend(ncol=3, framealpha=0.9, loc="upper right", columnspacing=0.8, handlelength=1.5)
    ax.set_ylim(0, 0.50)
    ax.set_xlim(0, 150)
    ax.grid(True)
    fig.savefig(FIG_DIR / "fig1_srd_convergence.pdf")
    plt.close()
    print("  Fig 1: SR/d convergence ✓")


# =============================================================================
# Figure 2: α dynamics — reversal vs no-reversal
# =============================================================================
def fig2_alpha_reversal():
    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.15, 2.8))

    models = [
        ("pythia_v2/pythia_70m.jsonl", "Pythia-70M (D/N=4261)", PALETTE_MAIN[0], '-'),
        ("pythia_v2/pythia_1b.jsonl", "Pythia-1B (D/N=297)", PALETTE_MAIN[1], '-'),
        ("pythia_v2/pythia_2.8b.jsonl", "Pythia-2.8B (D/N=108)", PALETTE_MAIN[2], '-'),
        ("pythia_v2/pythia_6.9b.jsonl", "Pythia-6.9B (D/N=44)", PALETTE_MAIN[3], '-'),
        ("olmo2_v2/olmo2_13b.jsonl", "OLMo-2-13B (D/N=365)", PALETTE_MAIN[4], '--'),
    ]

    for path_str, label, color, ls in models:
        path = RESULTS_DIR / path_str
        if not path.exists():
            continue
        records = load_jsonl(path)
        if not records or len(records) < 2:
            continue
        max_step = records[-1]["step"]
        if max_step == 0:
            continue
        frac = [r["step"] / max_step * 100 for r in records]
        alpha = [r["alpha_mean"] for r in records]
        ax.plot(frac, alpha, label=label, color=color, linestyle=ls, linewidth=1.4)

    ax.axhspan(2, 4, alpha=0.06, color='green', zorder=0)
    ax.text(70, 2.3, "Heavy-tail (well-trained)", fontsize=7, color='green', alpha=0.7)

    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel(r"$\alpha$ (power-law exponent)")
    ax.legend(fontsize=6.5, loc="right", framealpha=0.9)
    ax.set_ylim(1.5, 10)
    ax.set_xlim(0, 100)
    ax.grid(True)
    fig.savefig(FIG_DIR / "fig2_alpha_reversal.pdf")
    plt.close()
    print("  Fig 2: α reversal ✓")


# =============================================================================
# Figure 3: 3-Way schedule comparison (full width, 2 panels)
# =============================================================================
def fig3_3way_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8))

    schedules = {
        "cosine": ("Cosine", PALETTE_MAIN[0]),
        "wsd": ("WSD", PALETTE_MAIN[1]),
        "alpha": (r"$\alpha$-Guided", PALETTE_MAIN[2]),
    }

    for sched_key, (label, color) in schedules.items():
        all_losses = {}
        all_alphas = {}
        for seed in [42, 123]:
            path = RESULTS_DIR / "real_3way" / f"{sched_key}_s{seed}.log"
            if not path.exists():
                continue
            records = load_log(path)
            for r in records:
                if "loss" in r:
                    all_losses.setdefault(r["step"], []).append(r["loss"])
                if "alpha" in r:
                    all_alphas.setdefault(r["step"], []).append(r["alpha"])

        if all_losses:
            steps = sorted(all_losses.keys())
            mean_loss = [np.mean(all_losses[s]) for s in steps]
            axes[0].plot(np.array(steps)/1000, mean_loss, color=color, label=label, linewidth=1.4)

        if all_alphas:
            steps = sorted(all_alphas.keys())
            mean_alpha = [np.mean(all_alphas[s]) for s in steps]
            axes[1].plot(np.array(steps)/1000, mean_alpha, color=color, label=label, linewidth=1.8)

    # Zoom into the region where differences are visible
    axes[0].set_xlabel("Step (×10³)")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True)
    axes[0].set_title("(a) Training Loss", fontsize=9)
    axes[0].set_ylim(2.7, 4.5)  # zoom in to show tail differences
    axes[0].set_xlim(0, 9.5)

    axes[1].set_xlabel("Step (×10³)")
    axes[1].set_ylabel(r"$\alpha$")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True)
    axes[1].set_title(r"(b) Spectral Structure ($\alpha$)", fontsize=9)
    axes[1].set_ylim(2.2, 4.0)  # zoom in
    axes[1].set_xlim(0, 9.5)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG_DIR / "fig3_3way_comparison.pdf")
    plt.close()
    print("  Fig 3: 3-way comparison ✓")


# =============================================================================
# Figure 4: SR/d vs d — asymptotic model (single column)
# =============================================================================
def fig4_srd_vs_d():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.8))

    data = [
        (512, 0.074, "GPT-NeoX", "70M"),
        (768, 0.054, "GPT-NeoX", "160M"),
        (1024, 0.056, "GPT-NeoX", "410M"),
        (2048, 0.050, "GPT-NeoX", "1B"),
        (2560, 0.052, "GPT-NeoX", "2.8B"),
        (4096, 0.046, "GPT-NeoX", "6.9B"),
        (4096, 0.057, "LLaMA", "Amber"),
        (2048, 0.064, "OLMo2", "1B"),
        (4096, 0.046, "OLMo2", "7B"),
        (5120, 0.043, "OLMo2", "13B"),
        (5120, 0.043, "OLMo2", "32B"),
    ]

    arch_colors = {"GPT-NeoX": PALETTE_MAIN[0], "LLaMA": PALETTE_MAIN[1], "OLMo2": PALETTE_MAIN[2]}
    arch_markers = {"GPT-NeoX": "o", "LLaMA": "s", "OLMo2": "^"}

    for arch in ["GPT-NeoX", "LLaMA", "OLMo2"]:
        pts = [(d, sr) for d, sr, a, _ in data if a == arch]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                      c=arch_colors[arch], marker=arch_markers[arch], s=45,
                      label=arch, zorder=5, edgecolors='white', linewidths=0.5)

    d_range = np.linspace(400, 9000, 200)
    sr_pred = 0.040 + 0.61 / np.sqrt(d_range)
    ax.plot(d_range, sr_pred, color=ACCENT, linestyle='-', linewidth=1.2,
            label=r"$0.040 + 0.61/\sqrt{d}$", zorder=3)
    ax.axhline(y=0.040, color=GRAY, linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_xlabel(r"Hidden Dimension $d$")
    ax.set_ylabel(r"SR/$d$ (final)")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(True)
    ax.set_xlim(0, 6000)
    ax.set_ylim(0.03, 0.08)
    fig.savefig(FIG_DIR / "fig4_srd_vs_d.pdf")
    plt.close()
    print("  Fig 4: SR/d vs d ✓")


# =============================================================================
# Figure 5: MLP vs Attention α (grouped bar)
# =============================================================================
def fig5_mlp_attn_gap():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))

    models = ["2.8B", "6.9B", "13B", "32B", "65B"]
    attn = [4.71, 4.99, 6.25, 3.44, 4.50]
    mlp = [5.16, 5.13, 7.94, 7.59, 5.89]
    gaps = [m - a for m, a in zip(mlp, attn)]

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, attn, width, label=r'$\alpha_{\mathrm{attn}}$',
                   color=sns.color_palette("Blues_d", 3)[1], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, mlp, width, label=r'$\alpha_{\mathrm{mlp}}$',
                   color=sns.color_palette("Reds_d", 3)[1], edgecolor='white', linewidth=0.5)

    for i, gap in enumerate(gaps):
        ax.annotate(f"Δ={gap:.1f}", (i, max(attn[i], mlp[i]) + 0.15),
                   ha='center', fontsize=7, color='#333', fontstyle='italic')

    ax.axhline(y=4.0, color=GRAY, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_xlabel("Model Size")
    ax.set_ylabel(r"$\alpha$")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis='y')
    ax.set_ylim(0, 9.5)
    fig.savefig(FIG_DIR / "fig5_mlp_attn_gap.pdf")
    plt.close()
    print("  Fig 5: MLP/Attn gap ✓")


# =============================================================================
# Figure 6: Structural Chinchilla (log-scale x-axis)
# =============================================================================
def fig6_structural_chinchilla():
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.58, 3.4))

    # Data: (D/N, alpha, label, N_params, group)
    # group: "small" = N<=1B, "large" = N>1.7B
    data = [
        (4261, 2.60, "70M", 7e7, "small"),
        (1848, 2.63, "160M", 1.6e8, "small"),
        (740, 2.73, "410M", 4.1e8, "small"),
        (297, 2.78, "1B", 1e9, "small"),
        (108, 5.16, "2.8B", 2.8e9, "large"),
        (44, 5.13, "6.9B", 6.9e9, "large"),
        (187, 5.25, "Amber", 6.7e9, "large"),
        (365, 6.95, "13B", 13e9, "large"),
        (189, 5.25, "32B", 32e9, "large"),
        (21, 5.09, "K2-65B", 65e9, "large"),
    ]

    # Plot points: blue for small, red for large
    color_small = '#2166ac'
    color_large = '#b2182b'
    for dn, alpha, label, n, group in data:
        c = color_small if group == "small" else color_large
        marker = 'o' if group == "small" else 's'
        ax.scatter(dn, alpha, color=c, marker=marker, s=50, zorder=5,
                   edgecolors='white', linewidths=0.4)

    # Annotations — no overlaps
    offsets = {
        "70M": (6, 3), "160M": (6, 3), "410M": (6, 3),
        "1B": (6, 3), "2.8B": (6, 4), "6.9B": (6, -9),
        "Amber": (6, 4), "13B": (6, 3),
        "32B": (6, -9), "K2-65B": (6, -9),
    }
    for dn, alpha, label, n, group in data:
        ox, oy = offsets[label]
        c = color_small if group == "small" else color_large
        ax.annotate(label, (dn, alpha), fontsize=6, textcoords="offset points",
                    xytext=(ox, oy), color=c, fontweight='medium')

    # Original exponential curve (fits small models well)
    x = np.logspace(1, 4.2, 200)
    y = 2.54 + 3.5 * np.exp(-x / 269)
    ax.plot(x, y, color=color_small, linewidth=1.4, linestyle='-', alpha=0.8,
            label=r"Eq.~11: $\alpha = 2.54 + 3.5\,e^{-D/(269N)}$")

    # Horizontal band for large models
    ax.axhspan(4.9, 5.4, xmin=0, xmax=0.6, alpha=0.08, color=color_large, zorder=0)

    # Dashed line showing large-model mean (excluding 13B outlier)
    large_mean = np.mean([5.16, 5.13, 5.25, 5.25, 5.09])
    ax.axhline(y=large_mean, xmin=0, xmax=0.58, color=color_large, linewidth=1.0,
               linestyle='--', alpha=0.6)

    # Zone labels
    ax.text(7000, 2.3, r"$N \leq 1$B" + "\n(structurally mature)", fontsize=6.5,
            color=color_small, ha='center', va='center', alpha=0.8)
    ax.text(30, 5.55, r"$N > 1.7$B" + "\n(structurally immature)", fontsize=6.5,
            color=color_large, ha='left', va='bottom', alpha=0.8)

    # Arrow highlighting the key comparison: same D/N, different outcome
    ax.annotate("", xy=(320, 6.7), xytext=(320, 3.0),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=0.7,
                                connectionstyle='arc3,rad=0'))
    ax.text(380, 4.8, r"same $D/N$," + "\n" + r"different $N$",
            fontsize=5.5, color='#444', va='center', style='italic')

    # Heavy-tail threshold
    ax.axhline(y=4.0, color=GRAY, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.text(10000, 4.1, r"$\alpha\!=\!4$", fontsize=5.5, color=GRAY)

    ax.set_xlabel(r"$D/N$ (tokens per parameter)")
    ax.set_ylabel(r"$\alpha_{\mathrm{final}}$")
    ax.set_xscale('log')
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.95)
    ax.grid(True)
    ax.set_ylim(1.8, 7.5)
    ax.set_xlim(10, 12000)
    fig.savefig(FIG_DIR / "fig6_structural_chinchilla.pdf")
    plt.close()
    print("  Fig 6: Structural Chinchilla ✓")


# =============================================================================
if __name__ == "__main__":
    print("Generating paper figures (NeurIPS style)...")
    fig1_srd_convergence()
    fig2_alpha_reversal()
    fig3_3way_comparison()
    fig4_srd_vs_d()
    fig5_mlp_attn_gap()
    fig6_structural_chinchilla()
    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Format: PDF, 300 DPI, serif font, NeurIPS-compatible sizing")
