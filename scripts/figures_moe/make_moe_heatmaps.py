"""MoE per-layer spectral heatmaps.

Two figure groups:
  A. OLMoE training dynamics: layer (y) x checkpoint (x), one panel per metric
     (alpha, SR/d, EPR, psi, router SR/d). MoE-specific view dense models lack.
  B. Cross-model per-layer comparison: OLMoE vs Mixtral vs Phi-3.5 for alpha & SR/d.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path("docs/presentation/figures_moe")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load OLMoE training trajectory
# ---------------------------------------------------------------------------
olmoe = []
with open("results/olmoe_moe/olmoe_1b_7b.jsonl") as f:
    for line in f:
        olmoe.append(json.loads(line))
olmoe.sort(key=lambda d: d["step"])

steps = [d["step"] for d in olmoe]
n_layers = len(olmoe[0]["per_layer_summary"])
n_ckpt = len(olmoe)

# Build [layer, checkpoint] matrices
metrics = ["alpha_mean", "srd_mean", "epr", "psi_mean", "router_srd"]
labels = {
    "alpha_mean": r"$\alpha$ (power-law exponent)",
    "srd_mean": "SR/d (stable rank ratio)",
    "epr": "EPR (energy equipartition)",
    "psi_mean": r"$\psi$ (order parameter)",
    "router_srd": "Router SR/d",
}
mats = {m: np.zeros((n_layers, n_ckpt)) for m in metrics}
for j, d in enumerate(olmoe):
    for i, L in enumerate(sorted(d["per_layer_summary"], key=lambda x: x["layer"])):
        for m in metrics:
            mats[m][i, j] = L.get(m, np.nan)

# ---------------------------------------------------------------------------
# Figure A: OLMoE training dynamics heatmaps (5 panels)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
xt = [f"{s//1000}k" for s in steps]
for ax, m in zip(axes, metrics):
    im = ax.imshow(mats[m], aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(labels[m], fontsize=12)
    ax.set_xlabel("Training step")
    ax.set_xticks(range(0, n_ckpt, 2))
    ax.set_xticklabels(xt[::2], rotation=45, ha="right", fontsize=8)
    if ax is axes[0]:
        ax.set_ylabel("Layer index")
    ax.set_yticks(range(0, n_layers, 2))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("OLMoE-1B-7B per-layer spectral dynamics (16 layers x 10 checkpoints)", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "moe_heatmap_olmoe_dynamics.png", dpi=160)
print(f"Saved {OUT/'moe_heatmap_olmoe_dynamics.png'}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure A2: alpha & SR/d only, larger, with annotations (final checkpoint focus)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, m, cmap in zip(axes, ["alpha_mean", "srd_mean"], ["magma", "viridis"]):
    im = ax.imshow(mats[m], aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(labels[m], fontsize=13)
    ax.set_xlabel("Training step")
    ax.set_xticks(range(n_ckpt))
    ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Layer index")
    ax.set_yticks(range(n_layers))
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("OLMoE: alpha & SR/d across layers and training", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "moe_heatmap_olmoe_alpha_srd.png", dpi=160)
print(f"Saved {OUT/'moe_heatmap_olmoe_alpha_srd.png'}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure B: cross-model per-layer comparison (final checkpoint)
# ---------------------------------------------------------------------------
cross = []
with open("results/moe_cross_model/phase2_results.jsonl") as f:
    for line in f:
        cross.append(json.loads(line))

# OLMoE final checkpoint per-layer (all 16 layers)
olmoe_final = olmoe[-1]
model_layers = {}
short = {
    "mistralai/Mixtral-8x7B-v0.1": "Mixtral-8x7B\n(inter=14336)",
    "microsoft/Phi-3.5-MoE-instruct": "Phi-3.5-MoE\n(inter=6400)",
}
model_layers["OLMoE-1B-7B\n(inter=1024)"] = sorted(
    olmoe_final["per_layer_summary"], key=lambda x: x["layer"]
)
for d in cross:
    name = short.get(d["model"], d["model"])
    model_layers[name] = sorted(d["layer_details"], key=lambda x: x["layer"])

# Two heatmaps side by side: alpha and SR/d. Each row = layer (normalized depth), col = model.
fig, axes = plt.subplots(1, 2, figsize=(11, 7))
model_names = list(model_layers.keys())
# Use 8 depth bins (Mixtral/Phi sampled 8 layers); resample OLMoE to 8 bins
n_bins = 8

def resample(layers, key, n_bins):
    vals = [L[key] for L in layers]
    idx = np.linspace(0, len(vals) - 1, n_bins).astype(int)
    return np.array([vals[i] for i in idx])

for ax, key, cmap, title in [
    ("alpha_mean", None, "magma", r"Expert $\alpha$ per (relative) layer"),
    ("srd_mean", None, "viridis", "Expert SR/d per (relative) layer"),
]:
    pass

for ax, (key, cmap, title) in zip(
    axes,
    [("alpha_mean", "magma", r"Expert $\alpha$"), ("srd_mean", "viridis", "Expert SR/d")],
):
    M = np.zeros((n_bins, len(model_names)))
    for j, name in enumerate(model_names):
        M[:, j] = resample(model_layers[name], key, n_bins)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Relative depth (shallow -> deep)")
    ax.set_yticks(range(n_bins))
    # annotate values
    for i in range(n_bins):
        for j in range(len(model_names)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Cross-model per-layer spectral comparison (final checkpoint)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "moe_heatmap_crossmodel.png", dpi=160)
print(f"Saved {OUT/'moe_heatmap_crossmodel.png'}")
plt.close(fig)

print("\nDone. 3 heatmap figures written to", OUT)
