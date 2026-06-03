"""Detailed per-layer heatmaps: attention vs FFN, per-projection, per-expert.

Inputs: results/perlayer_detail/{olmoe,mixtral}_detail.json
Each record: {layer, kind(attn/expert/router), proj, expert_id, alpha, srd, psi, ...}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/presentation/figures_moe")
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.load(open(f"results/perlayer_detail/{name}_detail.json"))


def layers_of(recs):
    return sorted({r["layer"] for r in recs})


# ===========================================================================
# Figure 1: Attention (q/k/v/o) vs FFN-expert -- per-layer alpha & SR/d
#   Like the dense heatmap: rows = component, cols = layer.
# ===========================================================================
def fig_attn_vs_ffn(data, tag):
    recs = data["records"]
    layers = layers_of(recs)
    li2col = {l: i for i, l in enumerate(layers)}

    # Rows: q_proj, k_proj, v_proj, o_proj (attn), then expert-mean per proj
    attn_projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
    expert_projs = sorted({r["proj"] for r in recs if r["kind"] == "expert"})
    row_labels = [f"attn:{p.split('_')[0]}" for p in attn_projs] + \
                 [f"ffn:{p}" for p in expert_projs] + ["router"]

    for metric, cmap, mlabel in [("alpha", "magma", r"$\alpha$"), ("srd", "viridis", "SR/d")]:
        M = np.full((len(row_labels), len(layers)), np.nan)
        # attn rows
        for ri, p in enumerate(attn_projs):
            for r in recs:
                if r["kind"] == "attn" and r["proj"] == p:
                    M[ri, li2col[r["layer"]]] = r[metric]
        # ffn expert rows (mean across experts)
        for ki, p in enumerate(expert_projs):
            ri = len(attn_projs) + ki
            for l in layers:
                vals = [r[metric] for r in recs
                        if r["kind"] == "expert" and r["proj"] == p and r["layer"] == l]
                if vals:
                    M[ri, li2col[l]] = np.mean(vals)
        # router row
        ri = len(row_labels) - 1
        for r in recs:
            if r["kind"] == "router":
                M[ri, li2col[r["layer"]]] = r[metric]

        fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.7), 5))
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_xlabel("Layer index")
        ax.set_title(f"{data['model'].split('/')[-1]}: {mlabel} — attention vs FFN per layer")
        for i in range(len(row_labels)):
            for j in range(len(layers)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="white", fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        fn = OUT / f"moe_perlayer_{tag}_attn_vs_ffn_{metric}.png"
        fig.savefig(fn, dpi=160)
        print("Saved", fn)
        plt.close(fig)


# ===========================================================================
# Figure 2: Per-expert variation within layers (OLMoE) -- expert x layer alpha
#   Shows which experts diverge spectrally.
# ===========================================================================
def fig_per_expert(data, tag, proj="gate_proj"):
    recs = [r for r in data["records"] if r["kind"] == "expert" and r["proj"] == proj]
    if not recs:
        # try fused / alternate names
        projs = sorted({r["proj"] for r in data["records"] if r["kind"] == "expert"})
        proj = projs[0]
        recs = [r for r in data["records"] if r["kind"] == "expert" and r["proj"] == proj]
    layers = layers_of(recs)
    experts = sorted({r["expert_id"] for r in recs})
    li2col = {l: i for i, l in enumerate(layers)}
    ei2row = {e: i for i, e in enumerate(experts)}

    for metric, cmap, mlabel in [("alpha", "magma", r"$\alpha$"), ("srd", "viridis", "SR/d")]:
        M = np.full((len(experts), len(layers)), np.nan)
        for r in recs:
            M[ei2row[r["expert_id"]], li2col[r["layer"]]] = r[metric]
        fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.7), max(5, len(experts) * 0.3)))
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_ylabel("Expert ID")
        ax.set_yticks(range(len(experts)))
        ax.set_yticklabels(experts, fontsize=7)
        ax.set_xlabel("Layer index")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_title(f"{data['model'].split('/')[-1]}: per-expert {mlabel} ({proj})")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        fn = OUT / f"moe_perexpert_{tag}_{metric}.png"
        fig.savefig(fn, dpi=160)
        print("Saved", fn)
        plt.close(fig)


# ===========================================================================
# Figure 3: Expert spectral spread per layer (how much experts differ)
#   For each layer: min/mean/max alpha across experts -> shows divergence depth.
# ===========================================================================
def fig_expert_spread(data, tag):
    recs = [r for r in data["records"] if r["kind"] == "expert"]
    layers = layers_of(recs)
    means, stds, mins, maxs = [], [], [], []
    for l in layers:
        vals = [r["alpha"] for r in recs if r["layer"] == l]
        means.append(np.mean(vals)); stds.append(np.std(vals))
        mins.append(np.min(vals)); maxs.append(np.max(vals))
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.6), 5))
    x = range(len(layers))
    ax.fill_between(x, mins, maxs, alpha=0.2, color="C0", label="min–max range")
    ax.errorbar(x, means, yerr=stds, fmt="o-", color="C0", capsize=3, label="mean ± std")
    ax.set_xticks(list(x)); ax.set_xticklabels(layers, fontsize=8)
    ax.set_xlabel("Layer index"); ax.set_ylabel(r"Expert $\alpha$")
    ax.set_title(f"{data['model'].split('/')[-1]}: per-layer expert α spread")
    ax.legend()
    fig.tight_layout()
    fn = OUT / f"moe_expertspread_{tag}.png"
    fig.savefig(fn, dpi=160)
    print("Saved", fn)
    plt.close(fig)


if __name__ == "__main__":
    olmoe = load("olmoe")
    mixtral = load("mixtral")

    fig_attn_vs_ffn(olmoe, "olmoe")
    fig_attn_vs_ffn(mixtral, "mixtral")

    fig_per_expert(olmoe, "olmoe")
    fig_per_expert(mixtral, "mixtral")

    fig_expert_spread(olmoe, "olmoe")
    fig_expert_spread(mixtral, "mixtral")

    print("\nDone.")
