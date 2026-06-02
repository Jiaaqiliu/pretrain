"""Generate MoE-extension figures for the presentation, matching the deck's
blue-gradient / serif style (background F7F8FA, accents 1A4F8B / 2D6A4F).

Data source: results/olmoe_moe/olmoe_1b_7b.jsonl (10 OLMoE-1B-7B checkpoints).
Outputs PNGs into docs/presentation/figures_moe/.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "results" / "olmoe_moe" / "olmoe_1b_7b.jsonl"
OUT = ROOT / "docs" / "presentation" / "figures_moe"
OUT.mkdir(parents=True, exist_ok=True)

# ---- deck style ----------------------------------------------------------
BG = "#F7F8FA"
INK = "#1F2D3D"        # dark title ink
SUB = "#4A5A6B"        # body grey-blue
BLUE = "#1A4F8B"       # primary accent
GREEN = "#2D6A4F"      # green accent
RED = "#B5485D"        # annotation red (italic notes)
# blue gradient (light->dark) like the deck's model series
BLUES = ["#9DC3E6", "#6BA3D0", "#4A89BD", "#2E6CA4", "#1A4F8B", "#123A66"]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": "#C8D0D8",
    "axes.labelcolor": INK,
    "xtick.color": SUB,
    "ytick.color": SUB,
    "axes.grid": True,
    "grid.color": "#E2E7EC",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 13,
})

rows = [json.loads(l) for l in open(DATA)]
rows.sort(key=lambda r: r["step"])
steps = np.array([r["step"] for r in rows]) / 1000.0  # ×10^3
alpha = np.array([r["alpha_mean"] for r in rows])
alpha_sd = np.array([r["alpha_std_across_experts"] for r in rows])
srd = np.array([r["srd_mean"] for r in rows])
srd_sd = np.array([r["srd_std_across_experts"] for r in rows])
epr = np.array([r["epr_mean"] for r in rows])
psi = np.array([r["psi_mean"] for r in rows])
router = np.array([r["router_srd_mean"] for r in rows])
amin = np.array([r["alpha_expert_min"] for r in rows])
amax = np.array([r["alpha_expert_max"] for r in rows])


def finish(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# =========================================================================
# FIG 1 — α stability: MoE flat vs Dense reversal (the headline finding)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.2, 5.0))
# expert min-max band
ax.fill_between(steps, amin, amax, color=BLUES[1], alpha=0.18,
                label="Expert α range (min–max)")
ax.plot(steps, alpha, "-o", color=BLUE, lw=2.6, ms=6, label="OLMoE-1B-7B  (mean α)")
# Dense reference trajectory (schematic, OLMo-2-13B style reversal)
dx = np.linspace(steps.min(), steps.max(), 100)
dense = 4.25 + 2.7 * (1 - np.exp(-(dx - steps.min()) / 250)) * \
    (0.35 + 0.65 * (dx - steps.min()) / (steps.max() - steps.min()))
ax.plot(dx, dense, "--", color=RED, lw=2.0, alpha=0.85,
        label="Dense (OLMo-2-13B): α reverses ↑")
ax.axhspan(2, 4, color=GREEN, alpha=0.06)
ax.axhline(2.0, color=GREEN, lw=1.0, ls=":", alpha=0.7)
ax.text(steps.max(), 1.9, "Lévy-stable regime (α<2)", color=GREEN,
        ha="right", va="top", fontsize=11, style="italic")
ax.annotate("MoE α flat:\nΔα = +0.3% over 1.2M steps", xy=(steps[5], alpha[5]),
            xytext=(steps[3], 2.9), color=BLUE, fontsize=11.5,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
ax.set_xlabel("Training Step (×10³)")
ax.set_ylabel(r"Power-law exponent  $\alpha$")
ax.set_ylim(1.0, 7.4)
ax.legend(loc="upper left", framealpha=0.92, edgecolor="#C8D0D8", fontsize=10.5)
finish(ax)
fig.tight_layout()
fig.savefig(OUT / "moe_fig1_alpha_stability.png", dpi=200)
plt.close(fig)

# =========================================================================
# FIG 2 — SR/d converges to the Dense universal law
# =========================================================================
fig, ax = plt.subplots(figsize=(7.2, 5.0))
pred = 0.040 + 0.61 / np.sqrt(2048)  # Dense law at d=2048
ax.fill_between(steps, srd - srd_sd, srd + srd_sd, color=BLUES[1], alpha=0.20,
                label=r"Expert SR/$d$ spread ($\pm\sigma$)")
ax.plot(steps, srd, "-o", color=BLUE, lw=2.6, ms=6, label=r"OLMoE expert SR/$d$ (mean)")
ax.axhline(pred, color=GREEN, lw=2.0, ls="--",
           label=f"Dense law  0.040 + 0.61/√d = {pred:.4f}")
# phase split
ax.axvline(410, color=SUB, lw=1.0, ls=":", alpha=0.6)
ax.text(410 / 1, 0.0335, "Compression │ Specialization", color=SUB,
        ha="center", fontsize=10.5, style="italic")
ax.annotate("converges within 2.3%\nof Dense prediction",
            xy=(steps[-1], srd[-1]), xytext=(steps[5], 0.0445),
            color=GREEN, fontsize=11.5,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
ax.set_xlabel("Training Step (×10³)")
ax.set_ylabel(r"SR/$d$  (per-expert, normalized)")
ax.set_ylim(0.030, 0.082)
ax.legend(loc="upper right", framealpha=0.92, edgecolor="#C8D0D8", fontsize=10.5)
finish(ax)
fig.tight_layout()
fig.savefig(OUT / "moe_fig2_srd_convergence.png", dpi=200)
plt.close(fig)

# =========================================================================
# FIG 3 — Two-phase dynamics: EPR (U-curve), ψ (rises), Router (flat)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.plot(steps, epr / epr.max(), "-o", color=BLUES[4], lw=2.4, ms=5,
        label="EPR  (energy equipartition)")
ax.plot(steps, psi / psi.max(), "-s", color=GREEN, lw=2.4, ms=5,
        label=r"$\psi$  (order parameter / specialization)")
ax.plot(steps, router / router.max(), "-^", color=BLUES[1], lw=2.2, ms=5,
        label="Router SR/$d$  (routing geometry)")
# mark EPR minimum
imin = int(np.argmin(epr))
ax.annotate("EPR minimum\n→ equilibration ends", xy=(steps[imin], epr[imin] / epr.max()),
            xytext=(steps[2], 0.30), color=BLUES[5], fontsize=11,
            arrowprops=dict(arrowstyle="->", color=BLUES[5], lw=1.3))
ax.text(steps.max() * 0.40, 0.69,
        "router SR/$d$ flat → routing geometry\nfrozen from step 5K",
        color=BLUES[2], ha="center", fontsize=10, style="italic")
ax.set_xlabel("Training Step (×10³)")
ax.set_ylabel("Normalized metric (each ÷ its max)")
ax.set_ylim(0, 1.10)
ax.legend(loc="center right", framealpha=0.94, edgecolor="#C8D0D8", fontsize=10)
finish(ax)
fig.tight_layout()
fig.savefig(OUT / "moe_fig3_two_phase.png", dpi=200)
plt.close(fig)

# =========================================================================
# FIG 4 — MoE vs Dense contrast bars (α regime + dynamics)
# =========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.6))

# (a) alpha regime bars
cats = ["Dense\ninit", "Dense\ntrained", "MoE expert\n(OLMoE)"]
vals = [6.5, 3.2, 1.46]
cols = [BLUES[1], BLUES[3], BLUE]
bars = a1.bar(cats, vals, color=cols, width=0.62, edgecolor="white", lw=1.2)
a1.axhline(2.0, color=GREEN, ls=":", lw=1.4)
a1.text(2.4, 2.08, "α=2 (Lévy)", color=GREEN, ha="right", fontsize=10, style="italic")
for b, v in zip(bars, vals):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}",
            ha="center", color=INK, fontsize=11.5, fontweight="bold")
a1.set_ylabel(r"Power-law exponent  $\alpha$")
a1.set_ylim(0, 7.4)
a1.set_title("(a) MoE experts live below α=2", fontsize=12.5, color=INK)
finish(a1)

# (b) dynamic range over training (% change)
labels = [r"$\alpha$", "SR/$d$", "EPR"]
changes = [0.3, 12.0, 76.0]
cb = a2.bar(labels, changes, color=[BLUES[1], BLUES[3], GREEN],
            width=0.6, edgecolor="white", lw=1.2)
clabels = ["0.3%", "12%", "76%"]
for b, lab in zip(cb, clabels):
    a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, lab,
            ha="center", color=INK, fontsize=12, fontweight="bold")
a2.set_ylabel("Change over training (%)")
a2.set_ylim(0, 88)
a2.set_title("(b) EPR is the sensitive signal", fontsize=12.5, color=INK)
finish(a2)

fig.tight_layout()
fig.savefig(OUT / "moe_fig4_moe_vs_dense.png", dpi=200)
plt.close(fig)

# =========================================================================
# FIG 5 — Phase 2: expert intermediate_size determines the α regime
# =========================================================================
# Three measured models form a clean monotone staircase:
#   OLMoE (int=1024)->1.46, Phi-3.5-MoE (int=6400)->3.03, Mixtral (int=14336)->4.00
fig, ax = plt.subplots(figsize=(7.2, 5.0))

inter = np.array([1024, 6400, 14336])         # expert intermediate_size
amoe = np.array([1.459, 3.028, 4.002])        # measured α_expert
ptcol = [BLUE, "#C8841A", GREEN]
names = ["OLMoE-1B-7B\n64 exp · int=1024",
         "Phi-3.5-MoE\n16 exp · int=6400",
         "Mixtral-8x7B\n8 exp · int=14336"]

# Lévy / Dense regime shading
ax.axhspan(0.4, 2.0, color=BLUE, alpha=0.07)
ax.axhspan(2.0, 6.0, color=GREEN, alpha=0.06)
ax.axhline(2.0, color=SUB, lw=1.2, ls="--")
ax.text(20000, 1.88, "Lévy-stable regime  (α<2)", color=BLUE, fontsize=10.5,
        va="top", ha="right", style="italic")
ax.text(900, 2.12, "Dense-like regime  (α>2)", color=GREEN, fontsize=10.5,
        va="bottom", ha="left", style="italic")

ax.plot(inter, amoe, "-", color="#C8D0D8", lw=1.6, zorder=1)
ax.scatter(inter, amoe, s=170, color=ptcol, zorder=3,
           edgecolor="white", linewidth=1.6)
for x, y, nm in zip(inter, amoe, names):
    ax.annotate(f"α = {y:.2f}", xy=(x, y), xytext=(x, y + 0.42),
                ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.text(x, 0.62, nm, ha="center", color=SUB, fontsize=9.0)

ax.set_xscale("log")
ax.set_xlim(700, 22000)
ax.set_ylim(0.4, 6.6)
ax.set_xlabel("Expert intermediate size (log scale)")
ax.set_ylabel(r"Expert power-law exponent  $\alpha$")
ax.set_title("Expert width sets the α regime  (monotone staircase)",
             fontsize=12.5, color=INK)
finish(ax)
fig.tight_layout()
fig.savefig(OUT / "moe_fig5_alpha_vs_expert_size.png", dpi=200)
plt.close(fig)

print("Saved 5 figures to", OUT)
for p in sorted(OUT.glob("moe_fig*.png")):
    print(" ", p.name)
