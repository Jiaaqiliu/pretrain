"""Two figures from the new dense per-layer data (with psi/entropy):

  A. Unified α-vs-width staircase: dense Pythia (5 widths) + MoE per-expert
     widths on one axis. Shows alpha rises with matrix width in BOTH families.
  B. Dense per-layer psi & entropy heatmaps (block x checkpoint), the metrics
     we just back-filled so dense aligns with the MoE measurement set.
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

DENSE = [("70m", 512), ("410m", 1024), ("1b", 2048), ("2.8b", 2560), ("6.9b", 4096)]


def load_dense(sz):
    rows = [json.loads(l) for l in open(f"results/heatmap_v2/pythia_{sz}_perlayer.jsonl")]
    rows.sort(key=lambda r: r["step"])
    return rows


# ===========================================================================
# A. Unified alpha-vs-width staircase (dense MLP + MoE experts)
# ===========================================================================
def fig_unified_width():
    # dense: x = hidden_dim (proxy for MLP matrix width), y = mean MLP alpha
    dx, dy_mlp, dy_attn = [], [], []
    for sz, d in DENSE:
        r = load_dense(sz)[-1]
        dy_mlp.append(np.mean([L["alpha"] for L in r["layers"] if L["type"] == "mlp"]))
        dy_attn.append(np.mean([L["alpha"] for L in r["layers"] if L["type"] == "attn"]))
        dx.append(d)

    # MoE: x = intermediate_size (expert width), y = mean expert alpha
    moe_pts = []  # (intermediate, alpha, label)
    for fn, label in [("results/perlayer_detail/olmoe_detail.json", "OLMoE"),
                      ("results/perlayer_detail/mixtral_detail.json", "Mixtral")]:
        d = json.load(open(fn))
        recs = [x["alpha"] for x in d["records"] if x["kind"] == "expert"]
        moe_pts.append((d["intermediate_size"], np.nanmean(recs), label))
    # Phi-3.5 from phase2 aggregate
    for line in open("results/moe_cross_model/phase2_results.jsonl"):
        d = json.loads(line)
        if "Phi-3.5" in d["model"]:
            moe_pts.append((d["intermediate_size"], d["alpha_moe"], "Phi-3.5"))

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.plot(dx, dy_mlp, "o-", color="#B2182B", label="Dense MLP (Pythia)", ms=8)
    ax.plot(dx, dy_attn, "s--", color="#2171B5", label="Dense attention (Pythia)",
            ms=7, alpha=0.8)
    for inter, a, label in moe_pts:
        ax.scatter([inter], [a], s=170, marker="*", zorder=5,
                   edgecolor="k", linewidth=0.6,
                   color={"OLMoE": "#6A51A3", "Phi-3.5": "#C8841A",
                          "Mixtral": "#2D6A4F"}.get(label, "grey"))
        ax.annotate(f"{label}\n(int={inter})", (inter, a),
                    textcoords="offset points", xytext=(8, -4), fontsize=8.5)
    ax.axhline(2.0, ls=":", color="grey", lw=1)
    ax.text(520, 2.06, "Lévy boundary α=2", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("Matrix width proxy  (dense: hidden_dim · MoE: expert intermediate_size)")
    ax.set_ylabel(r"mean $\alpha$")
    ax.set_title("Same qualitative trend in both families: wider matrices → higher α\n"
                 "(axes use different width proxies; compare trends, not absolute overlap)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fn = OUT / "unified_alpha_vs_width.png"
    fig.savefig(fn, dpi=160)
    print("Saved", fn)
    plt.close(fig)


# ===========================================================================
# B. Dense psi & entropy heatmaps (block x checkpoint), MLP rows
# ===========================================================================
def block_idx(name):
    m = BLOCK_RE.search(name)
    return int(m.group(1)) if m else -1


def fig_psi_entropy(sz, model_label):
    rows = load_dense(sz)
    steps = [r["step"] for r in rows]
    blocks = sorted({block_idx(L["name"]) for L in rows[0]["layers"]
                     if L["type"] == "mlp" and block_idx(L["name"]) >= 0})
    bi = {b: i for i, b in enumerate(blocks)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    specs = [
        ("psi", "cividis", r"MLP order parameter $\psi$"),
        ("entropy", "viridis", "MLP spectral entropy (nats)"),
    ]
    for ax, (metric, cmap, title) in zip(axes, specs):
        M = np.full((len(blocks), len(rows)), np.nan)
        for j, r in enumerate(rows):
            acc = {}
            for L in r["layers"]:
                if L["type"] == "mlp" and block_idx(L["name"]) in bi:
                    acc.setdefault(block_idx(L["name"]), []).append(L.get(metric, np.nan))
            for b, v in acc.items():
                M[bi[b], j] = np.nanmean(v)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Training step")
        nt = max(1, len(steps) // 8)
        ax.set_xticks(range(0, len(steps), nt))
        ax.set_xticklabels([f"{steps[i]//1000}k" for i in range(0, len(steps), nt)],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Transformer block")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{model_label}: per-block ψ & entropy dynamics (back-filled metrics)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fn = OUT / f"dense_psi_entropy_{sz}.png"
    fig.savefig(fn, dpi=150)
    print("Saved", fn)
    plt.close(fig)


if __name__ == "__main__":
    fig_unified_width()
    fig_psi_entropy("1b", "Pythia-1B")
    fig_psi_entropy("6.9b", "Pythia-6.9B")
    print("\nDone.")
