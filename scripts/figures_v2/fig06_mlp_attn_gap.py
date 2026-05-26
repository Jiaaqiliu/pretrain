"""
Figure 6: MLP vs Attention α Gap (Lollipop Chart)

Shows the growing structural gap between attention (which matures) and
MLP (which stays random) as model size increases.

Data: Hardcoded from measurements
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, ax = plt.subplots(figsize=(COL_W * 1.15, 3.0))

    # Data: (name, alpha_attn, alpha_mlp, architecture)
    # Note: Pythia-6.9B has attn > mlp (unusual), kept for completeness
    models_data = [
        ('Pythia\n2.8B', 4.79, 5.53, 'GPT-NeoX'),
        ('Pythia\n6.9B', 5.16, 5.13, 'GPT-NeoX'),
        ('Amber\n7B', 4.53, 5.87, 'LLaMA'),
        ('K2\n65B', 4.50, 5.89, 'LLaMA'),
        ('OLMo-2\n13B', 6.25, 7.94, 'OLMo2'),
        ('OLMo-2\n32B', 3.44, 7.59, 'OLMo2'),
        ('Mistral\n7B', 3.79, 9.22, 'Mistral'),
    ]

    # Sort by gap
    models_data.sort(key=lambda x: x[2] - x[1])
    names = [m[0] for m in models_data]
    attn = [m[1] for m in models_data]
    mlp = [m[2] for m in models_data]
    archs = [m[3] for m in models_data]

    x = np.arange(len(names))

    # Connecting lines (gap bars)
    for i in range(len(names)):
        ax.plot([x[i], x[i]], [attn[i], mlp[i]], color=C['gray_pale'],
                linewidth=3.0, alpha=0.5, zorder=1, solid_capstyle='round')

    # Attention dots
    ax.scatter(x, attn, color=C['blue'], s=60, zorder=5, marker='o',
               edgecolors='white', linewidths=0.6, label=r'$\alpha_{\mathrm{attn}}$')
    # MLP dots
    ax.scatter(x, mlp, color=C['red'], s=60, zorder=5, marker='s',
               edgecolors='white', linewidths=0.6, label=r'$\alpha_{\mathrm{mlp}}$')

    # Gap annotations
    for i in range(len(names)):
        gap = abs(mlp[i] - attn[i])
        mid = (attn[i] + mlp[i]) / 2
        ha = 'left'
        ax.text(x[i] + 0.22, mid, f'Δ={gap:.1f}', fontsize=5.5,
                color=C['gray'], va='center', ha=ha, style='italic')

    # Heavy-tail threshold
    ax.axhline(4.0, color=C['green'], ls='--', lw=0.8, alpha=0.4)
    ax.text(0.3, 3.5, r'Heavy-tail ($\alpha\!<\!4$)', fontsize=5.5,
            color=C['green'], alpha=0.6, ha='left', va='top')

    # Random threshold
    ax.axhline(6.0, color=C['red_light'], ls='--', lw=0.8, alpha=0.4)
    ax.text(0.3, 6.3, r'Random ($\alpha\!>\!6$)', fontsize=5.5,
            color=C['red_light'], alpha=0.6, ha='left')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel(r'$\alpha$ (power-law exponent)')
    ax.set_ylim(2.5, 10.5)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.92)
    add_subtle_grid(ax, axis='y')

    save_fig(fig, 'fig06_mlp_attn_gap')


if __name__ == '__main__':
    main()
