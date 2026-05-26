"""
Figure 4: 3-Way Schedule Comparison (Loss + α + Downstream)

Three panels comparing Cosine vs WSD vs α-Guided on 410M model:
(a) Training loss over time
(b) α evolution during training
(c) Downstream benchmark bar chart

Data: results/real_3way/*.log + results/eval_410m/summary.json
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.5))

    # --- Load training logs ---
    schedule_data = {}
    for sched in ['cosine', 'wsd', 'alpha']:
        all_losses = {}
        all_alphas = {}
        for seed in [42, 123]:
            path = RESULTS / 'real_3way' / f'{sched}_s{seed}.log'
            if not path.exists():
                continue
            records = load_log(path)
            for r in records:
                if 'loss' in r:
                    all_losses.setdefault(r['step'], []).append(r['loss'])
                if 'alpha' in r:
                    all_alphas.setdefault(r['step'], []).append(r['alpha'])
        schedule_data[sched] = {'losses': all_losses, 'alphas': all_alphas}

    # === Panel A: Training Loss ===
    for sched in ['cosine', 'wsd', 'alpha']:
        data = schedule_data[sched]
        if not data['losses']:
            continue
        steps = sorted(data['losses'].keys())
        mean_loss = [np.mean(data['losses'][s]) for s in steps]

        # Confidence band (if 2 seeds)
        if all(len(data['losses'][s]) >= 2 for s in steps):
            std_loss = [np.std(data['losses'][s]) for s in steps]
            steps_arr = np.array(steps) / 1000
            axes[0].fill_between(steps_arr,
                                np.array(mean_loss) - np.array(std_loss),
                                np.array(mean_loss) + np.array(std_loss),
                                alpha=0.08, color=SCHED[sched]['color'])

        axes[0].plot(np.array(steps)/1000, mean_loss,
                    color=SCHED[sched]['color'], ls=SCHED[sched]['ls'],
                    label=SCHED[sched]['label'], linewidth=1.4)

    axes[0].set_xlabel('Step (×10³)')
    axes[0].set_ylabel('Training Loss')
    axes[0].legend(fontsize=6.5, loc='upper right')
    axes[0].set_xlim(0, 9.5)
    axes[0].set_ylim(2.7, 4.5)
    add_subtle_grid(axes[0])
    axes[0].set_title('(a) Training Loss', fontsize=8.5, pad=5)

    # === Panel B: α Dynamics ===
    for sched in ['cosine', 'wsd', 'alpha']:
        data = schedule_data[sched]
        if not data['alphas']:
            continue
        steps = sorted(data['alphas'].keys())
        mean_alpha = [np.mean(data['alphas'][s]) for s in steps]
        axes[1].plot(np.array(steps)/1000, mean_alpha,
                    color=SCHED[sched]['color'], ls=SCHED[sched]['ls'],
                    label=SCHED[sched]['label'], linewidth=1.6)

    axes[1].set_xlabel('Step (×10³)')
    axes[1].set_ylabel(r'$\alpha$')
    axes[1].legend(fontsize=6.5, loc='upper right')
    axes[1].set_xlim(0, 9.5)
    axes[1].set_ylim(2.2, 4.2)
    add_subtle_grid(axes[1])
    axes[1].set_title(r'(b) Spectral Structure ($\alpha$)', fontsize=8.5, pad=5)

    # === Panel C: Downstream Benchmarks ===
    benchmarks = ['ARC-E', 'Hella\nSwag', 'LAM\nBADA', 'PIQA', 'Wino\nGrande']
    cosine_scores = [0.550, 0.307, 0.284, 0.645, 0.510]
    wsd_scores = [0.567, 0.314, 0.293, 0.659, 0.504]
    alpha_scores = [0.574, 0.313, 0.302, 0.655, 0.498]

    x = np.arange(len(benchmarks))
    w = 0.24

    axes[2].bar(x - w, cosine_scores, w, color=SCHED['cosine']['color'],
                alpha=0.75, edgecolor='white', linewidth=0.4, label='Cosine')
    axes[2].bar(x, wsd_scores, w, color=SCHED['wsd']['color'],
                alpha=0.75, edgecolor='white', linewidth=0.4, label='WSD')
    axes[2].bar(x + w, alpha_scores, w, color=SCHED['alpha']['color'],
                alpha=0.75, edgecolor='white', linewidth=0.4, label=r'$\alpha$-Guided')

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(benchmarks, fontsize=5.5)
    axes[2].set_ylabel('Accuracy (0-shot)')
    axes[2].set_ylim(0.24, 0.70)
    axes[2].legend(fontsize=5.5, loc='upper left', ncol=1)
    add_subtle_grid(axes[2], axis='y')
    axes[2].set_title('(c) Downstream Benchmarks', fontsize=8.5, pad=5)

    # Average score annotations
    avgs = [0.459, 0.467, 0.468]
    labels_avg = ['Cos', 'WSD', r'$\alpha$']
    for i, (avg, lab) in enumerate(zip(avgs, labels_avg)):
        axes[2].text(4.6, 0.64 - i*0.035, f'{lab}: {avg:.3f}',
                    fontsize=5.5, color=list(SCHED.values())[i]['color'],
                    family='monospace', va='top')

    fig.tight_layout(w_pad=1.8)
    save_fig(fig, 'fig04_3way_comparison')


if __name__ == '__main__':
    main()
