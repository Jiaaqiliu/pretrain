"""
Figure 3: α Dynamics with MLP/Attention Decomposition

Panel A: Overall α trajectories for representative models (showing reversal)
Panel B: MLP vs Attention α for OLMo-2-13B (showing the gap and reversal)

Data: results/pythia_v2/*.jsonl + results/olmo2_v2/olmo2_13b.jsonl
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.8))

    # === Panel A: Overall α dynamics ===
    models = [
        ('pythia_v2/pythia_70m.jsonl', 'Pythia-70M', 512, MODEL_GRADIENT[0], '-'),
        ('pythia_v2/pythia_1b.jsonl', 'Pythia-1B', 2048, MODEL_GRADIENT[3], '-'),
        ('pythia_v2/pythia_6.9b.jsonl', 'Pythia-6.9B', 4096, MODEL_GRADIENT[5], '-'),
        ('olmo2_v2/olmo2_13b.jsonl', 'OLMo-2-13B', 5120, C['orange'], '--'),
    ]

    for path_str, label, d, color, ls in models:
        path = RESULTS / path_str
        if not path.exists():
            continue
        records = load_jsonl(path)
        if len(records) < 2:
            continue
        max_step = records[-1]['step']
        if max_step == 0:
            continue
        frac = np.array([r['step'] / max_step * 100 for r in records])
        alpha = np.array([r['alpha_mean'] for r in records])
        ax1.plot(frac, alpha, color=color, ls=ls, label=label, linewidth=1.4)

    # Heavy-tail zone
    ax1.axhspan(2, 4, alpha=0.04, color=C['green'], zorder=0)
    ax1.text(72, 3.4, 'Heavy-tail zone', fontsize=6, color=C['green'],
             alpha=0.6, style='italic')

    ax1.set_xlabel('Training Progress (%)')
    ax1.set_ylabel(r'$\alpha$ (power-law exponent)')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(1.5, 20)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_yticks([2, 3, 5, 8, 12, 18])
    ax1.legend(fontsize=6, loc='upper right', framealpha=0.92)
    add_subtle_grid(ax1)
    ax1.set_title('(a) Overall α trajectories', fontsize=8.5, pad=6)

    # === Panel B: MLP/Attn decomposition (OLMo-2-13B) ===
    olmo_path = RESULTS / 'olmo2_v2/olmo2_13b.jsonl'
    if olmo_path.exists():
        records = load_jsonl(olmo_path)
        max_step = records[-1]['step']
        frac = np.array([r['step'] / max_step * 100 for r in records])
        alpha_attn = np.array([r['alpha_attn'] for r in records])
        alpha_mlp = np.array([r['alpha_mlp'] for r in records])

        # Main lines
        ax2.plot(frac, alpha_attn, color=C['blue'], linewidth=1.6,
                label=r'$\alpha_{\mathrm{attn}}$', zorder=3)
        ax2.plot(frac, alpha_mlp, color=C['red'], linewidth=1.6,
                label=r'$\alpha_{\mathrm{mlp}}$', zorder=3)

        # Gap fill
        ax2.fill_between(frac, alpha_attn, alpha_mlp, alpha=0.06,
                         color=C['purple_light'], zorder=1)

        # Reversal annotation
        min_idx_attn = np.argmin(alpha_attn)
        if min_idx_attn > 0 and min_idx_attn < len(frac) - 2:
            rev_frac = frac[min_idx_attn]
            ax2.axvline(rev_frac, color=C['red'], ls=':', lw=0.8, alpha=0.5)
            ax2.annotate(r'$\alpha$ reversal', xy=(rev_frac, alpha_attn[min_idx_attn]),
                        xytext=(rev_frac + 12, alpha_attn[min_idx_attn] - 1.0),
                        fontsize=6, color=C['red'], style='italic',
                        arrowprops=dict(arrowstyle='->', color=C['red'],
                                       lw=0.7, connectionstyle='arc3,rad=0.2'))

            # Shade reversal region
            ax2.axvspan(rev_frac, 100, alpha=0.03, color=C['red_pale'], zorder=0)

        # Gap annotation
        mid_idx = len(frac) // 2
        gap = alpha_mlp[mid_idx] - alpha_attn[mid_idx]
        ax2.annotate(f'Gap = {gap:.1f}', xy=(frac[mid_idx], (alpha_attn[mid_idx] + alpha_mlp[mid_idx])/2),
                    xytext=(frac[mid_idx] + 15, (alpha_attn[mid_idx] + alpha_mlp[mid_idx])/2),
                    fontsize=6, color=C['purple'], ha='left',
                    arrowprops=dict(arrowstyle='-', color=C['purple_light'], lw=0.5))

    # Threshold
    ax2.axhline(4.0, color=C['gray_light'], ls=':', lw=0.7)
    ax2.text(3, 3.4, r'$\alpha\!=\!4$', fontsize=5.5, color=C['gray'])

    ax2.set_xlabel('Training Progress (%)')
    ax2.set_ylabel(r'$\alpha$')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(2, 21)
    ax2.legend(fontsize=7, loc='upper right', framealpha=0.92)
    add_subtle_grid(ax2)
    ax2.set_title('(b) OLMo-2-13B: MLP vs Attention', fontsize=8.5, pad=6)

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'fig03_alpha_dynamics')


if __name__ == '__main__':
    main()
