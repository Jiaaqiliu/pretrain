"""
Figure 9: Training Dynamics Dashboard

Annotated multi-panel time series showing how spectral monitoring works
in practice: Loss + LR, α dynamics with reversal detection, SR/d compression.

Data: results/real_3way/alpha_s42.log
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(COL_W * 1.3, 4.5),
                                         sharex=True)

    path = RESULTS / 'real_3way' / 'alpha_s42.log'
    if not path.exists():
        print('  ⚠ Missing: results/real_3way/alpha_s42.log')
        plt.close(fig)
        return

    records = load_log(path)

    # Extract data
    loss_data = [(r['step']/1000, r['loss']) for r in records if 'loss' in r]
    lr_data = [(r['step']/1000, r['lr']) for r in records if 'lr' in r]
    alpha_data = [(r['step']/1000, r['alpha']) for r in records if 'alpha' in r]
    sr_data = [(r['step']/1000, r['sr_d']) for r in records if 'sr_d' in r]

    # Find reversal point
    if alpha_data:
        alpha_vals = [a[1] for a in alpha_data]
        alpha_steps = [a[0] for a in alpha_data]
        min_idx = np.argmin(alpha_vals)
        reversal_step = alpha_steps[min_idx] if min_idx < len(alpha_steps) - 1 else None
    else:
        reversal_step = None

    max_step = max(r['step']/1000 for r in records) if records else 9

    # === Panel 1: Loss + LR ===
    if loss_data:
        steps, losses = zip(*loss_data)
        ax1.plot(steps, losses, color=C['navy'], linewidth=1.2, label='Loss')
    ax1.set_ylabel('Loss', color=C['navy'], fontsize=7.5)
    ax1.tick_params(axis='y', labelcolor=C['navy'])

    if lr_data:
        ax1_r = ax1.twinx()
        steps, lrs = zip(*lr_data)
        ax1_r.plot(steps, lrs, color=C['orange_light'], linewidth=0.9, alpha=0.7, label='LR')
        ax1_r.set_ylabel('LR', color=C['orange'], fontsize=7)
        ax1_r.tick_params(axis='y', labelcolor=C['orange'], labelsize=6)
        ax1_r.spines['right'].set_visible(True)
        ax1_r.spines['right'].set_linewidth(0.4)
        ax1_r.spines['right'].set_color(C['orange_light'])

    add_subtle_grid(ax1)
    ax1.set_title(r'$\alpha$-Guided Training Monitor (Pythia-410M)', fontsize=8.5, pad=6)

    # === Panel 2: α with reversal detection ===
    if alpha_data:
        steps_a, alphas = zip(*alpha_data)
        ax2.plot(steps_a, alphas, color=C['red'], linewidth=1.6,
                marker='o', markersize=3.5, markerfacecolor=C['red'],
                markeredgecolor='white', markeredgewidth=0.4)

        if reversal_step and reversal_step < max_step * 0.85:
            # Reversal marker
            ax2.axvline(reversal_step, color=C['red'], ls='--', lw=0.9, alpha=0.6)
            ax2.annotate('Reversal detected\n→ Start LR decay',
                        xy=(reversal_step, alpha_vals[min_idx]),
                        xytext=(reversal_step + 0.8, alpha_vals[min_idx] + 0.3),
                        fontsize=5.5, color=C['red'],
                        arrowprops=dict(arrowstyle='->', color=C['red'], lw=0.7),
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=C['red_pale'],
                                 edgecolor='none', alpha=0.3))
        else:
            # No reversal — model stays healthy (410M is small enough)
            ax2.text(0.95, 0.85, '● No reversal\n(monotonic ↓)',
                    transform=ax2.transAxes, fontsize=6, color=C['green'],
                    ha='right', va='top', fontweight='medium')

    ax2.set_ylabel(r'$\alpha$', color=C['red'], fontsize=8)
    ax2.tick_params(axis='y', labelcolor=C['red'])
    add_subtle_grid(ax2)

    # === Panel 3: SR/d ===
    if sr_data:
        steps_s, sr_vals = zip(*sr_data)
        ax3.plot(steps_s, sr_vals, color=C['blue'], linewidth=1.6,
                marker='s', markersize=3, markerfacecolor=C['blue'],
                markeredgecolor='white', markeredgewidth=0.4)

    ax3.set_ylabel(r'SR/$d$', color=C['blue'], fontsize=8)
    ax3.set_xlabel('Step (×10³)')
    ax3.tick_params(axis='y', labelcolor=C['blue'])
    add_subtle_grid(ax3)

    # Zone coloring across all panels
    if reversal_step and reversal_step < max_step * 0.95:
        for a in [ax1, ax2, ax3]:
            a.axvspan(0, reversal_step, alpha=0.02, color=C['green'],
                     zorder=0, label='_nolegend_')
            a.axvspan(reversal_step, max_step, alpha=0.02, color=C['red_pale'],
                     zorder=0, label='_nolegend_')

        # Phase labels
        ax1.text(reversal_step * 0.4, ax1.get_ylim()[1] * 0.95,
                '● HEALTHY', fontsize=6, color=C['green'], va='top', fontweight='bold')
        ax1.text((reversal_step + max_step) / 2, ax1.get_ylim()[1] * 0.95,
                '● DECAY', fontsize=6, color=C['red'], va='top', fontweight='bold')

    fig.tight_layout(h_pad=0.3)
    save_fig(fig, 'fig09_dashboard')


if __name__ == '__main__':
    main()
