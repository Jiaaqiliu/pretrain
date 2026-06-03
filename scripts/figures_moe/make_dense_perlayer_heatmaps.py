"""Dense per-layer heatmaps from existing results/heatmap/*.jsonl.

Two figure groups (mirroring the MoE per-layer figures so they can sit
side-by-side in the deck / paper):

  A. Training dynamics: block (y) x checkpoint (x) for alpha & SR/d.
     Shows WHERE (which depth) and WHEN (which step) structure forms.
  B. MLP vs Attention per-block: the classic "MLP lags attention" pattern,
     the dense counterpart to the MoE alpha_attn < alpha_ffn reversal.
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/presentation/figures_moe")
OUT.mkdir(parents=True, exist_ok=True)

BLOCK_RE = re.compile(r"layers\.(\d+)\.")


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda d: d["step"])
    return rows


def block_idx(name):
    m = BLOCK_RE.search(name)
    return int(m.group(1)) if m else -1


def comp_of(name):
    """Return q/k/v-ish component label for a weight name."""
    if "query_key_value" in name:
        return "attn:qkv"
    if "attention.dense" in name:
        return "attn:out"
    if "h_to_4h" in name:
        return "mlp:up"
    if "4h_to_h" in name:
        return "mlp:down"
    return None


# ---------------------------------------------------------------------------
# Build [block, checkpoint] matrices for a metric, averaging attn or mlp blocks
# ---------------------------------------------------------------------------
def build_dyn_matrix(rows, kind, metric):
    """kind in {'attn','mlp'}. Returns (matrix[block,ckpt], steps, n_blocks)."""
    steps = [r["step"] for r in rows]
    # discover blocks from first ckpt
    blocks = sorted({block_idx(L["name"]) for L in rows[0]["layers"]
                     if L["type"] == kind and block_idx(L["name"]) >= 0})
    bi = {b: i for i, b in enumerate(blocks)}
    M = np.full((len(blocks), len(rows)), np.nan)
    for j, r in enumerate(rows):
        acc = {}
        for L in r["layers"]:
            if L["type"] != kind:
                continue
            b = block_idx(L["name"])
            if b < 0:
                continue
            acc.setdefault(b, []).append(L[metric])
        for b, vals in acc.items():
            M[bi[b], j] = np.mean(vals)
    return M, steps, blocks


def fig_dynamics(rows, tag, model_label):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    combos = [
        ("attn", "alpha", "magma", r"Attention $\alpha$"),
        ("mlp", "alpha", "magma", r"MLP $\alpha$"),
        ("attn", "sr_d", "viridis", "Attention SR/d"),
        ("mlp", "sr_d", "viridis", "MLP SR/d"),
    ]
    for ax, (kind, metric, cmap, title) in zip(axes.flat, combos):
        M, steps, blocks = build_dyn_matrix(rows, kind, metric)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Training step")
        nt = max(1, len(steps) // 8)
        ax.set_xticks(range(0, len(steps), nt))
        ax.set_xticklabels([f"{steps[i]//1000}k" for i in range(0, len(steps), nt)],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Transformer block")
        ax.set_yticks(range(0, len(blocks), max(1, len(blocks)//8)))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{model_label}: per-block spectral dynamics (attention vs MLP)",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = OUT / f"dense_dynamics_{tag}.png"
    fig.savefig(fn, dpi=150)
    print("Saved", fn)
    plt.close(fig)


def fig_mlp_vs_attn(rows, tag, model_label):
    """Final-checkpoint per-block comparison: 4 component rows x blocks."""
    r = rows[-1]
    comps = ["attn:qkv", "attn:out", "mlp:up", "mlp:down"]
    blocks = sorted({block_idx(L["name"]) for L in r["layers"]
                     if block_idx(L["name"]) >= 0})
    bi = {b: i for i, b in enumerate(blocks)}

    for metric, cmap, mlabel in [("alpha", "magma", r"$\alpha$"),
                                 ("sr_d", "viridis", "SR/d")]:
        M = np.full((len(comps), len(blocks)), np.nan)
        for L in r["layers"]:
            c = comp_of(L["name"])
            b = block_idx(L["name"])
            if c in comps and b in bi:
                M[comps.index(c), bi[b]] = L[metric]
        fig, ax = plt.subplots(figsize=(max(9, len(blocks) * 0.32), 3.6))
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels(comps, fontsize=10)
        ax.set_xlabel("Transformer block")
        step = max(1, len(blocks) // 16)
        ax.set_xticks(range(0, len(blocks), step))
        ax.set_xticklabels([blocks[i] for i in range(0, len(blocks), step)], fontsize=8)
        ax.set_title(f"{model_label}: {mlabel} — MLP vs Attention per block (final ckpt)",
                     fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fn = OUT / f"dense_mlp_vs_attn_{tag}_{metric}.png"
        fig.savefig(fn, dpi=150)
        print("Saved", fn)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Combined contrast: dense (MLP>>attn) vs MoE (attn<ffn) — final ckpt means
# ---------------------------------------------------------------------------
def fig_dense_vs_moe_contrast(dense_rows, dense_label):
    r = dense_rows[-1]
    attn_a = np.mean([L["alpha"] for L in r["layers"] if L["type"] == "attn"])
    mlp_a = np.mean([L["alpha"] for L in r["layers"] if L["type"] == "mlp"])

    # MoE numbers from per-layer detail
    moe = json.load(open("results/perlayer_detail/olmoe_detail.json"))
    recs = moe["records"]
    moe_attn = np.nanmean([x["alpha"] for x in recs if x["kind"] == "attn"])
    moe_ffn = np.nanmean([x["alpha"] for x in recs if x["kind"] == "expert"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    groups = [f"Dense\n({dense_label})", "MoE\n(OLMoE-1B-7B)"]
    attn_vals = [attn_a, moe_attn]
    ffn_vals = [mlp_a, moe_ffn]
    gap_d = mlp_a - attn_a
    gap_m = moe_ffn - moe_attn
    x = np.arange(len(groups))
    w = 0.35
    b1 = ax.bar(x - w/2, attn_vals, w, label="Attention", color="#2171B5")
    b2 = ax.bar(x + w/2, ffn_vals, w, label="MLP / FFN-expert", color="#B2182B")
    ax.axhline(2.0, ls="--", color="grey", lw=1, label=r"Lévy boundary $\alpha=2$")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel(r"mean $\alpha$")
    ax.set_title("Same ordering (FFN > attn), but MoE pushed into Lévy region\n"
                 f"and the gap shrinks: dense Δα={gap_d:.2f} → MoE Δα={gap_m:.2f}",
                 fontsize=11.5)
    ax.legend(fontsize=9)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fn = OUT / "dense_vs_moe_attn_ffn.png"
    fig.savefig(fn, dpi=160)
    print("Saved", fn)
    plt.close(fig)


if __name__ == "__main__":
    p1 = load("results/heatmap/pythia_1b_perlayer.jsonl")
    p7 = load("results/heatmap/pythia_6.9b_perlayer.jsonl")

    fig_dynamics(p1, "pythia1b", "Pythia-1B")
    fig_dynamics(p7, "pythia6.9b", "Pythia-6.9B")

    fig_mlp_vs_attn(p1, "pythia1b", "Pythia-1B")
    fig_mlp_vs_attn(p7, "pythia6.9b", "Pythia-6.9B")

    fig_dense_vs_moe_contrast(p1, "Pythia-1B")

    print("\nDone.")
