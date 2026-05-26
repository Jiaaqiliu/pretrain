"""
Figure 11: Cross-Architecture Validation Grid

2×2 panel grid showing SR/d and α patterns are universal across
GPT-NeoX (Pythia) and OLMo-2 architectures.

Data: results/pythia_v2/*.jsonl + results/olmo2_v2/*.jsonl
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W * 0.78, 4.5))

    # === Top-left: GPT-NeoX SR/d ===
    ax = axes[0, 0]
    subset = [0, 2, 3, 5]  # 70M, 410M, 1B, 6.9B
    for idx in subset:
        fname, label, d = PYTHIA_MODELS[idx]
        path = RESULTS / 'pythia_v2' / f'{fname}.jsonl'
        if not path.exists():
            continue
        records = load_jsonl(path)
        max_step = records[-1]['step'] if records else 1
        frac = [r['step'] / max_step * 100 for r in records]
        sr_d = [r['stable_rank_mean'] / d for r in records]
        ax.plot(frac, sr_d, color=MODEL_GRADIENT[idx], linewidth=1.2,
                label=label, alpha=0.85)
    ax.set_ylabel(r'SR/$d$')
    ax.set_ylim(0, 0.48)
    ax.set_title('GPT-NeoX (Pythia)', fontsize=8, color=C['blue'], fontweight='medium')
    ax.legend(fontsize=5.5, ncol=2, loc='upper right')
    add_subtle_grid(ax)

    # === Top-right: OLMo-2 SR/d ===
    ax = axes[0, 1]
    for i, (fname, label, d) in enumerate(OLMO_MODELS):
        path = RESULTS / 'olmo2_v2' / f'{fname}.jsonl'
        if not path.exists():
            continue
        records = load_jsonl(path)
        max_step = records[-1]['step'] if records else 1
        frac = [r['step'] / max_step * 100 for r in records]
        sr_d = [r['stable_rank_mean'] / d for r in records]
        ax.plot(frac, sr_d, color=OLMO_COLORS[i], linewidth=1.2,
                label=label, alpha=0.85)
    ax.set_ylim(0, 0.48)
    ax.set_title('OLMo-2', fontsize=8, color=C['orange'], fontweight='medium')
    ax.legend(fontsize=5.5, ncol=2, loc='upper right')
    add_subtle_grid(ax)

    # === Bottom-left: GPT-NeoX α ===
    ax = axes[1, 0]
    for idx in subset:
        fname, label, d = PYTHIA_MODELS[idx]
        path = RESULTS / 'pythia_v2' / f'{fname}.jsonl'
        if not path.exists():
            continue
        records = load_jsonl(path)
        max_step = records[-1]['step'] if records else 1
        frac = [r['step'] / max_step * 100 for r in records]
        alpha = [r['alpha_mean'] for r in records]
        ax.plot(frac, alpha, color=MODEL_GRADIENT[idx], linewidth=1.2,
                label=label, alpha=0.85)
    ax.set_ylabel(r'$\alpha$')
    ax.set_xlabel('Training Progress (%)')
    ax.set_ylim(1.5, 20)
    ax.axhline(4.0, color=C['gray_light'], ls=':', lw=0.5)
    ax.set_title(r'GPT-NeoX: $\alpha$ dynamics', fontsize=8, color=C['blue'], fontweight='medium')
    ax.legend(fontsize=5.5, ncol=2, loc='upper right')
    add_subtle_grid(ax)

    # === Bottom-right: OLMo-2 α ===
    ax = axes[1, 1]
    for i, (fname, label, d) in enumerate(OLMO_MODELS):
        path = RESULTS / 'olmo2_v2' / f'{fname}.jsonl'
        if not path.exists():
            continue
        records = load_jsonl(path)
        max_step = records[-1]['step'] if records else 1
        frac = [r['step'] / max_step * 100 for r in records]
        alpha = [r['alpha_mean'] for r in records]
        ax.plot(frac, alpha, color=OLMO_COLORS[i], linewidth=1.2,
                label=label, alpha=0.85)
    ax.set_xlabel('Training Progress (%)')
    ax.set_ylim(1.5, 20)
    ax.axhline(4.0, color=C['gray_light'], ls=':', lw=0.5)
    ax.set_title(r'OLMo-2: $\alpha$ dynamics', fontsize=8, color=C['orange'], fontweight='medium')
    ax.legend(fontsize=5.5, ncol=2, loc='upper right')
    add_subtle_grid(ax)

    # Common formatting
    for a in axes.flat:
        a.set_xlim(0, 100)

    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save_fig(fig, 'fig11_cross_arch')


if __name__ == '__main__':
    main()
