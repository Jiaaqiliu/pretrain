"""
Figure 1: Spectral Phase Portrait (HERO FIGURE)

2D trajectory in (SR/d, α) space showing how models evolve during training.
All trajectories start at upper-right (random, high entropy) and converge
toward lower-left (compressed, structured) — but large models stall.

Data: results/pythia_v2/*.jsonl + results/olmo2_v2/olmo2_13b.jsonl
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, ax = plt.subplots(figsize=(TEXT_W * 0.58, 4.0))

    # --- Pythia suite (6 models) ---
    pythia_data = [
        ('pythia_v2/pythia_70m.jsonl', '70M', 512, MODEL_GRADIENT[0]),
        ('pythia_v2/pythia_160m.jsonl', '160M', 768, MODEL_GRADIENT[1]),
        ('pythia_v2/pythia_410m.jsonl', '410M', 1024, MODEL_GRADIENT[2]),
        ('pythia_v2/pythia_1b.jsonl', '1B', 2048, MODEL_GRADIENT[3]),
        ('pythia_v2/pythia_2.8b.jsonl', '2.8B', 2560, MODEL_GRADIENT[4]),
        ('pythia_v2/pythia_6.9b.jsonl', '6.9B', 4096, MODEL_GRADIENT[5]),
    ]

    # Store endpoints for later labeling
    endpoints = []

    for path_str, label, d, color in pythia_data:
        path = RESULTS / path_str
        if not path.exists():
            print(f'  ⚠ Missing: {path}')
            continue
        records = load_jsonl(path)
        if len(records) < 3:
            continue

        sr_d = np.array([r['stable_rank_mean'] / d for r in records])
        alpha = np.array([r['alpha_mean'] for r in records])

        # Main trajectory
        ax.plot(sr_d, alpha, color=color, linewidth=1.5, alpha=0.85, zorder=3)

        # Start marker (open circle)
        ax.plot(sr_d[0], alpha[0], 'o', color=color, markersize=3.5,
                markerfacecolor='white', markeredgewidth=0.8, zorder=5)

        # End marker (filled)
        ax.plot(sr_d[-1], alpha[-1], 'o', color=color, markersize=5,
                markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.5, zorder=5)

        endpoints.append((sr_d[-1], alpha[-1], label, color))

    # --- Label endpoints with arrows pointing to clear space ---
    # Place all small-model labels in a vertical stack to the RIGHT of the cluster
    # This avoids overlap entirely
    label_positions = {
        # (text_x, text_y) in data coordinates — placed in clear space
        '70M':  (0.11, 2.3),
        '160M': (0.11, 2.55),
        '410M': (0.11, 2.82),
        '1B':   (0.11, 3.1),
        '2.8B': (0.09, 5.6),
        '6.9B': (0.09, 4.7),
    }

    for sx, sy, label, color in endpoints:
        if label in label_positions:
            tx, ty = label_positions[label]
            ax.annotate(label, xy=(sx, sy), xytext=(tx, ty),
                       fontsize=6.5, color=color, fontweight='medium',
                       ha='left', va='center',
                       arrowprops=dict(arrowstyle='-', color=color,
                                      lw=0.5, alpha=0.5,
                                      connectionstyle='arc3,rad=0.1'))

    # --- OLMo-2-13B (shows reversal) ---
    olmo_path = RESULTS / 'olmo2_v2/olmo2_13b.jsonl'
    if olmo_path.exists():
        records = load_jsonl(olmo_path)
        sr_d = np.array([r['stable_rank_mean'] / 5120 for r in records])
        alpha = np.array([r['alpha_mean'] for r in records])

        ax.plot(sr_d, alpha, color=C['orange'], linewidth=1.6, alpha=0.85,
                linestyle='--', zorder=3)
        ax.plot(sr_d[0], alpha[0], 'o', color=C['orange'], markersize=3.5,
                markerfacecolor='white', markeredgewidth=0.8, zorder=5)
        ax.plot(sr_d[-1], alpha[-1], '^', color=C['orange'], markersize=6,
                markerfacecolor=C['orange'], markeredgecolor='white',
                markeredgewidth=0.5, zorder=5)

        # Label with arrow to clear space
        ax.annotate('OLMo-2-13B\n(reversal)', xy=(sr_d[-1], alpha[-1]),
                    xytext=(0.12, 7.5),
                    fontsize=6.5, color=C['orange'], fontweight='medium',
                    ha='left', va='center',
                    arrowprops=dict(arrowstyle='->', color=C['orange'],
                                   lw=0.7, connectionstyle='arc3,rad=-0.2'))

    # --- Zone shading (subtle, no text overlap) ---
    ax.axhspan(2, 4, alpha=0.035, color=C['green'], zorder=0)
    ax.axhspan(4, 6, alpha=0.02, color=C['gold'], zorder=0)

    # Zone labels placed at far right where no data exists
    ax.text(0.42, 2.9, 'Mature', fontsize=6, color=C['green'],
            alpha=0.6, ha='right', style='italic')
    ax.text(0.42, 4.8, 'Transition', fontsize=6, color=C['gold'],
            alpha=0.5, ha='right', style='italic')

    # --- Axes ---
    ax.set_xlabel(r'SR/$d$ (spectral compression $\rightarrow$)')
    ax.set_ylabel(r'$\alpha$ (power-law exponent)')
    ax.set_xlim(0.02, 0.48)
    ax.set_ylim(1.8, 22)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks([2, 3, 4, 5, 7, 10, 15, 20])
    add_subtle_grid(ax)

    # --- Legend (compact, upper right) ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=MODEL_GRADIENT[0], lw=1.5, label='Pythia (small)'),
        Line2D([0], [0], color=MODEL_GRADIENT[5], lw=1.5, label='Pythia (large)'),
        Line2D([0], [0], color=C['orange'], lw=1.5, ls='--', label='OLMo-2-13B'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor=C['gray'], markersize=4, label='Init'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=C['gray'],
               markeredgecolor='white', markersize=5, label='Final'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6.5,
              framealpha=0.95, handlelength=1.5, borderpad=0.4)

    save_fig(fig, 'fig01_phase_portrait')


if __name__ == '__main__':
    main()
