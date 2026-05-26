"""
Figure 2: SR/d Convergence Across Scales

Time-series showing stable rank normalized by hidden dimension converging
to a universal d-dependent constant. Includes inset zoom on final values.

Data: results/pythia_v2/*.jsonl
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, ax = plt.subplots(figsize=(COL_W * 1.1, 2.8))

    for i, (fname, label, d) in enumerate(PYTHIA_MODELS):
        path = RESULTS / 'pythia_v2' / f'{fname}.jsonl'
        if not path.exists():
            continue
        records = load_jsonl(path)
        steps = np.array([r['step'] for r in records]) / 1000
        sr_d = np.array([r['stable_rank_mean'] / d for r in records])
        ax.plot(steps, sr_d, color=MODEL_GRADIENT[i], label=label,
                linewidth=1.4, alpha=0.9)

    # Convergence band
    ax.axhspan(0.043, 0.074, alpha=0.05, color=C['red_pale'], zorder=0)
    ax.text(108, 0.078, r'Final SR/$d$ range [0.046–0.074]', fontsize=5.5,
            color=C['red'], alpha=0.7, style='italic')

    # Asymptotic limit
    ax.axhline(0.040, color=C['gray_light'], ls=':', lw=0.7, alpha=0.5)
    ax.text(2, 0.030, r'Asymptotic limit $c_0 = 0.040$', fontsize=5.5,
            color=C['gray'], alpha=0.6)

    ax.set_xlabel('Training Step (×10³)')
    ax.set_ylabel(r'SR/$d$')
    ax.set_xlim(0, 145)
    ax.set_ylim(0, 0.48)
    ax.legend(ncol=2, loc='upper right', fontsize=6.5,
              columnspacing=0.8, handlelength=1.5)
    add_subtle_grid(ax)

    # --- Inset: zoom on final convergence ---
    axins = ax.inset_axes([0.35, 0.32, 0.40, 0.38])
    for i, (fname, label, d) in enumerate(PYTHIA_MODELS):
        path = RESULTS / 'pythia_v2' / f'{fname}.jsonl'
        if not path.exists():
            continue
        records = load_jsonl(path)
        steps = np.array([r['step'] for r in records]) / 1000
        sr_d = np.array([r['stable_rank_mean'] / d for r in records])
        mask = steps > 70
        if mask.any():
            axins.plot(steps[mask], sr_d[mask], color=MODEL_GRADIENT[i], linewidth=1.1)

    axins.set_xlim(70, 145)
    axins.set_ylim(0.040, 0.095)
    axins.tick_params(labelsize=5.5)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    axins.spines['top'].set_linewidth(0.3)
    axins.spines['right'].set_linewidth(0.3)
    axins.axhline(0.040, color=C['gray_light'], ls=':', lw=0.5)
    add_subtle_grid(axins, alpha=0.08)
    ax.indicate_inset_zoom(axins, edgecolor=C['gray_light'], linewidth=0.5)

    save_fig(fig, 'fig02_srd_convergence')


if __name__ == '__main__':
    main()
