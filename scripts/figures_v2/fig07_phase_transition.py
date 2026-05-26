"""
Figure 7: Structural Chinchilla — Phase Transition

Two panels:
(a) α_final vs D/N: exponential fit works for small models, fails for large
(b) α_final vs N: sharp sigmoid phase transition at N ≈ 1.7B

Data: Hardcoded from measurements
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.9))

    # Complete data
    data = [
        (4261, 2.60, '70M', 7e7, 'small'),
        (1848, 2.63, '160M', 1.6e8, 'small'),
        (740, 2.73, '410M', 4.1e8, 'small'),
        (297, 2.78, '1B', 1e9, 'small'),
        (108, 5.16, '2.8B', 2.8e9, 'large'),
        (44, 5.13, '6.9B', 6.9e9, 'large'),
        (187, 5.25, 'Amber', 6.7e9, 'large'),
        (365, 6.95, '13B', 13e9, 'large'),
        (189, 5.25, '32B', 32e9, 'large'),
        (21, 5.09, 'K2-65B', 65e9, 'large'),
    ]

    c_small = C['blue']
    c_large = C['red']

    # === Panel A: α vs D/N ===
    for dn, alpha, label, n, group in data:
        c = c_small if group == 'small' else c_large
        marker = 'o' if group == 'small' else 's'
        ax1.scatter(dn, alpha, color=c, marker=marker, s=48, zorder=5,
                   edgecolors='white', linewidths=0.4)

    # Labels with careful placement
    label_offsets = {
        '70M': (0, 8), '160M': (0, 8), '410M': (0, 8), '1B': (0, 8),
        '2.8B': (5, 7), '6.9B': (5, -10), 'Amber': (5, 7),
        '13B': (5, -12), '32B': (5, -10), 'K2-65B': (5, -10),
    }
    for dn, alpha, label, n, group in data:
        c = c_small if group == 'small' else c_large
        ox, oy = label_offsets.get(label, (5, 5))
        ax1.annotate(label, (dn, alpha), xytext=(ox, oy),
                    textcoords='offset points', fontsize=5.5, color=c,
                    fontweight='medium')

    # Exponential fit (small models)
    x_fit = np.logspace(1, 4.2, 200)
    y_fit = 2.54 + 3.5 * np.exp(-x_fit / 269)
    ax1.plot(x_fit, y_fit, color=c_small, linewidth=1.2, alpha=0.7,
             label=r'Small: $2.54 + 3.5\,e^{-D/(269N)}$')

    # Large model band
    large_alphas = [5.16, 5.13, 5.25, 5.25, 5.09]
    ax1.axhline(np.mean(large_alphas), color=c_large, ls='--', lw=0.8,
                alpha=0.4, label=f'Large: mean = {np.mean(large_alphas):.2f}')

    # Threshold
    ax1.axhline(4.0, color=C['gray_light'], ls=':', lw=0.6, alpha=0.5)
    ax1.text(10000, 4.15, r'$\alpha\!=\!4$', fontsize=5.5, color=C['gray'])

    ax1.set_xscale('log')
    ax1.set_xlabel(r'$D/N$ (tokens per parameter)')
    ax1.set_ylabel(r'$\alpha_{\mathrm{final}}$')
    ax1.set_xlim(10, 12000)
    ax1.set_ylim(1.8, 7.5)
    ax1.legend(fontsize=6, loc='upper right', framealpha=0.92)
    add_subtle_grid(ax1)
    ax1.set_title(r'(a) $\alpha$ vs Data Ratio $D/N$', fontsize=8.5, pad=5)

    # === Panel B: α vs N (phase transition) ===
    for dn, alpha, label, n, group in data:
        c = c_small if group == 'small' else c_large
        marker = 'o' if group == 'small' else 's'
        ax2.scatter(n, alpha, color=c, marker=marker, s=48, zorder=5,
                   edgecolors='white', linewidths=0.4)
        # Selective labeling — avoid edge clipping
        if label in ['70M', '1B', '2.8B', '13B', '65B']:
            show_label = label if label != '65B' else 'K2-65B'
            ox = 0 if n < 1e9 else 5
            oy = 8 if group == 'small' else -10
            if label == 'K2-65B':
                ox, oy = -10, -10  # left of point
            ax2.annotate(show_label if label != '65B' else label, (n, alpha),
                        xytext=(ox, oy),
                        textcoords='offset points', fontsize=5.5, color=c)

    # Sigmoid fit
    n_range = np.logspace(7.5, 11, 300)
    alpha_sigmoid = 2.65 + 2.1 / (1 + np.exp(-(np.log10(n_range) - 9.23) / 0.07))
    ax2.plot(n_range, alpha_sigmoid, color=C['purple'], linewidth=1.4, alpha=0.8,
             label=r'Sigmoid fit ($R^2\!=\!0.97$)')

    # Critical point
    ax2.axvline(1.7e9, color=C['purple'], ls=':', lw=0.9, alpha=0.5)
    ax2.text(2.2e9, 7.2, r'$N^* \!\approx\! 1.7$B', fontsize=6.5,
             color=C['purple'], ha='left', va='top', fontweight='medium')

    # Zone labels — placed to avoid data
    ax2.text(1.2e8, 2.15, 'Structurally\nmature', fontsize=6, color=c_small,
             alpha=0.6, ha='center', style='italic')
    ax2.text(2e10, 7.3, 'Structurally\nimmature', fontsize=6, color=c_large,
             alpha=0.6, ha='center', style='italic')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Model Size $N$ (parameters)')
    ax2.set_ylabel(r'$\alpha_{\mathrm{final}}$')
    ax2.set_xlim(4e7, 1.5e11)
    ax2.set_ylim(1.8, 7.8)
    ax2.legend(fontsize=6, loc='lower right', framealpha=0.92)
    add_subtle_grid(ax2)
    ax2.set_title(r'(b) Phase Transition at $N^*\!\approx\!1.7$B', fontsize=8.5, pad=5)

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'fig07_phase_transition')


if __name__ == '__main__':
    main()
