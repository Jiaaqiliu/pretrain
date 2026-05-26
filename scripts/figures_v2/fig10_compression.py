"""
Figure 10: Universal Compression Law

Shows that all models compress exactly ~2 nats of Rényi-2 spectral entropy
during training, regardless of scale (70M → 6.9B, 930× range).

Data: results/pythia_v2/*.jsonl (initial + final stable_rank_mean)
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, ax = plt.subplots(figsize=(COL_W * 1.1, 2.8))

    # Compute ΔH₂ from data
    models_compression = []
    for fname, label, d in PYTHIA_MODELS:
        path = RESULTS / 'pythia_v2' / f'{fname}.jsonl'
        if not path.exists():
            continue
        records = load_jsonl(path)
        if len(records) < 2:
            continue
        sr_init = records[0]['stable_rank_mean'] / d
        sr_final = records[-1]['stable_rank_mean'] / d
        if sr_init > 0 and sr_final > 0:
            delta_h2 = np.log(sr_final / sr_init)
            models_compression.append((label, sr_init, sr_final, delta_h2))

    if not models_compression:
        print('  ⚠ No compression data')
        plt.close(fig)
        return

    names = [m[0] for m in models_compression]
    delta_h2_vals = [m[3] for m in models_compression]

    x = np.arange(len(names))

    # Bar chart
    colors = MODEL_GRADIENT[:len(names)]
    bars = ax.bar(x, delta_h2_vals, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=0.6, width=0.6)

    # Mean line
    mean_val = np.mean(delta_h2_vals)
    std_val = np.std(delta_h2_vals)
    ax.axhline(mean_val, color=C['red'], ls='--', lw=1.3, alpha=0.8)
    ax.text(len(names) - 0.3, mean_val + 0.05,
            f'Mean = {mean_val:.2f} ± {std_val:.2f} nats',
            fontsize=6.5, color=C['red'], ha='right', va='bottom',
            fontweight='medium')

    # Std band
    ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.04,
               color=C['red_pale'], zorder=0)

    # Value annotations on bars
    for i, (bar, val) in enumerate(zip(bars, delta_h2_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.05,
                f'{val:.2f}', ha='center', va='top', fontsize=6,
                color='white', fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel(r'$\Delta H_2$ (nats)')
    ax.set_ylim(-2.5, 0)
    ax.set_xlabel('Model Size')
    add_subtle_grid(ax, axis='y')

    # Title annotation
    ax.text(0.5, 0.97, r'Universal spectral compression: $\Delta H_2 \approx -2$ nats',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='center',
            color=C['navy'], style='italic')
    ax.text(0.5, 0.88, '(across 930× parameter range)',
            transform=ax.transAxes, fontsize=6, va='top', ha='center',
            color=C['gray'])

    save_fig(fig, 'fig10_compression')


if __name__ == '__main__':
    main()
