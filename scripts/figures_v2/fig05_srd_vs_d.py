"""
Figure 5: SR/d vs Hidden Dimension

Scatter plot showing the asymptotic formula SR/d = 0.040 + 0.61/√d
validated across 13 models from 4 architectures.

Data: Hardcoded from final measurements (all 13 models)
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, ax = plt.subplots(figsize=(COL_W * 1.1, 2.8))

    # All measured data points: (d, SR/d_final, architecture, label)
    data = [
        (512, 0.074, 'GPT-NeoX', '70M'),
        (768, 0.054, 'GPT-NeoX', '160M'),
        (1024, 0.056, 'GPT-NeoX', '410M'),
        (2048, 0.050, 'GPT-NeoX', '1B'),
        (2560, 0.052, 'GPT-NeoX', '2.8B'),
        (4096, 0.046, 'GPT-NeoX', '6.9B'),
        (4096, 0.057, 'LLaMA', 'Amber-7B'),
        (8192, 0.036, 'LLaMA', 'K2-65B'),
        (2048, 0.064, 'OLMo2', 'OLMo-1B'),
        (4096, 0.046, 'OLMo2', 'OLMo-7B'),
        (5120, 0.043, 'OLMo2', 'OLMo-13B'),
        (5120, 0.043, 'OLMo2', 'OLMo-32B'),
        (4096, 0.040, 'Mistral', 'Mistral-7B'),
    ]

    # Plot by architecture
    for arch, style in ARCH_STYLE.items():
        pts = [(d, sr) for d, sr, a, _ in data if a == arch]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                      c=style['color'], marker=style['marker'], s=50,
                      label=style['label'], zorder=5,
                      edgecolors='white', linewidths=0.5, alpha=0.9)

    # Asymptotic curve
    d_range = np.linspace(350, 9500, 300)
    sr_pred = 0.040 + 0.61 / np.sqrt(d_range)
    ax.plot(d_range, sr_pred, color=C['red'], linewidth=1.3, ls='-',
            label=r'$\mathrm{SR}/d = 0.040 + 0.61/\sqrt{d}$', zorder=3, alpha=0.8)

    # Confidence band (±10%)
    ax.fill_between(d_range, sr_pred * 0.85, sr_pred * 1.15,
                    alpha=0.04, color=C['red_pale'], zorder=1)

    # Asymptotic limit
    ax.axhline(0.040, color=C['gray_light'], ls=':', lw=0.7, alpha=0.6)
    ax.text(8800, 0.037, r'$c_0\!=\!0.040$', fontsize=6, color=C['gray'],
            ha='right', va='top')

    # Key annotations — avoid overlap with asymptotic line text
    ax.annotate('K2-65B', (8192, 0.036), xytext=(7500, 0.030),
               fontsize=5.5, color=C['green'], ha='center',
               arrowprops=dict(arrowstyle='-', color=C['gray_light'], lw=0.4))
    ax.annotate('13B ≡ 32B\n(same d)', (5120, 0.043), xytext=(6500, 0.055),
               fontsize=5.5, color=C['orange'], ha='center',
               arrowprops=dict(arrowstyle='-', color=C['gray_light'], lw=0.4))

    ax.set_xlabel(r'Hidden Dimension $d$')
    ax.set_ylabel(r'SR/$d$ (final)')
    ax.set_xlim(0, 9000)
    ax.set_ylim(0.025, 0.085)
    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.92)
    add_subtle_grid(ax)

    save_fig(fig, 'fig05_srd_vs_d')


if __name__ == '__main__':
    main()
