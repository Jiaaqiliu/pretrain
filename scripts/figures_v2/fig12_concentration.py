"""
Figure 12: Spectral Concentration Evolution

Shows how eigenvalue concentration (top-1, top-5, top-10) increases
during training — from uniform distribution toward spike structure.

Data: results/pythia_v2/*.jsonl (concentration_top1/5/10 fields)
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.4))

    metrics = [
        ('concentration_top1', '(a) Top-1 Concentration'),
        ('concentration_top5', '(b) Top-5 Concentration'),
        ('concentration_top10', '(c) Top-10 Concentration'),
    ]

    for panel_idx, (metric_key, title) in enumerate(metrics):
        ax = axes[panel_idx]

        for i, (fname, label, d) in enumerate(PYTHIA_MODELS):
            path = RESULTS / 'pythia_v2' / f'{fname}.jsonl'
            if not path.exists():
                continue
            records = load_jsonl(path)
            max_step = records[-1]['step'] if records else 1
            frac = [r['step'] / max_step * 100 for r in records]
            vals = [r.get(metric_key, 0) for r in records]
            ax.plot(frac, vals, color=MODEL_GRADIENT[i], linewidth=1.2,
                    label=label if panel_idx == 0 else '', alpha=0.85)

        ax.set_xlabel('Training Progress (%)')
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=7.5, pad=4)
        add_subtle_grid(ax)

    axes[0].set_ylabel('Concentration')
    axes[0].legend(fontsize=5.5, ncol=2, loc='upper left')

    fig.tight_layout(w_pad=1.5)
    save_fig(fig, 'fig12_concentration')


if __name__ == '__main__':
    main()
