"""
Figure 8: SR/d → Downstream Performance Correlation

Scatter plot showing SR/d predicts downstream benchmark scores
(Spearman r = -0.918, R² = 0.754).

Data: results/pythia_v2/*.jsonl + results/pythia_benchmarks/*.json
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *

def main():
    fig, ax = plt.subplots(figsize=(COL_W * 1.15, 3.0))

    benchmark_dir = RESULTS / 'pythia_benchmarks'
    spectral_dir = RESULTS / 'pythia_v2'

    if not benchmark_dir.exists():
        print('  ⚠ Missing: results/pythia_benchmarks/')
        plt.close(fig)
        return

    all_sr_d = []
    all_bench = []
    all_colors = []
    all_sizes = []

    for i, (model_id, label, d) in enumerate(PYTHIA_MODELS):
        # Load spectral data
        spec_path = spectral_dir / f'{model_id}.jsonl'
        if not spec_path.exists():
            continue
        spec_records = load_jsonl(spec_path)
        spec_by_step = {r['step']: r['stable_rank_mean'] / d for r in spec_records}

        # Load benchmark data
        model_short = label.lower()
        if model_id == 'pythia_70m':
            model_short = '70m'
        elif model_id == 'pythia_160m':
            model_short = '160m'
        elif model_id == 'pythia_410m':
            model_short = '410m'
        elif model_id == 'pythia_1b':
            model_short = '1b'
        elif model_id == 'pythia_2.8b':
            model_short = '2.8b'
        elif model_id == 'pythia_6.9b':
            model_short = '6.9b'

        for bm_file in sorted(benchmark_dir.glob(f'{model_short}_step*.json')):
            step_str = bm_file.stem.split('_step')[1]
            step = int(step_str)

            # Find nearest spectral measurement
            if step in spec_by_step:
                sr_d_val = spec_by_step[step]
            else:
                nearest = min(spec_by_step.keys(), key=lambda s: abs(s - step))
                if abs(nearest - step) > 10000:
                    continue
                sr_d_val = spec_by_step[nearest]

            try:
                with open(bm_file) as f:
                    bm_data = json.load(f)
                scores = []
                for task in ['lambada_openai', 'piqa', 'winogrande', 'arc_easy']:
                    if task in bm_data.get('results', {}):
                        acc = bm_data['results'][task].get('acc', None)
                        if acc is not None:
                            scores.append(acc)
                if len(scores) >= 3:
                    avg_score = np.mean(scores)
                    all_sr_d.append(sr_d_val)
                    all_bench.append(avg_score)
                    all_colors.append(MODEL_GRADIENT[i])
                    all_sizes.append(25 + i * 5)
            except (json.JSONDecodeError, KeyError):
                continue

    if not all_sr_d:
        print('  ⚠ No correlation data available')
        plt.close(fig)
        return

    all_sr_d = np.array(all_sr_d)
    all_bench = np.array(all_bench)

    # Scatter plot
    for i in range(len(all_sr_d)):
        ax.scatter(all_sr_d[i], all_bench[i], color=all_colors[i],
                  s=all_sizes[i], alpha=0.65, edgecolors='white',
                  linewidths=0.3, zorder=3)

    # Regression on log(SR/d)
    valid = (all_sr_d > 0) & (all_bench > 0)
    log_sr = np.log10(all_sr_d[valid])
    bench_valid = all_bench[valid]

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sr, bench_valid)
    x_line = np.linspace(log_sr.min(), log_sr.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(10**x_line, y_line, color=C['red'], linewidth=1.4, ls='-',
            zorder=4, alpha=0.8)

    # Confidence interval
    n = len(log_sr)
    se_line = std_err * np.sqrt(1/n + (x_line - log_sr.mean())**2 / ((log_sr - log_sr.mean())**2).sum())
    ax.fill_between(10**x_line, y_line - 1.96*se_line, y_line + 1.96*se_line,
                    alpha=0.06, color=C['red_pale'], zorder=2)

    # Statistics
    rho, p_spearman = stats.spearmanr(all_sr_d[valid], bench_valid)
    r2 = r_value**2

    stats_text = (f'Spearman ρ = {rho:.3f}\n'
                  f'R² = {r2:.3f}\n'
                  f'N = {n}')
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            fontsize=6.5, va='top', color=C['navy'], family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=C['gray_pale'], alpha=0.9))

    # Legend for model sizes
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], marker='o', color='none',
                        markerfacecolor=MODEL_GRADIENT[i], markersize=5,
                        label=PYTHIA_MODELS[i][1]) for i in range(6)]
    ax.legend(handles=legend_els, fontsize=6, loc='lower left',
              ncol=2, columnspacing=0.5, framealpha=0.92, title='Model',
              title_fontsize=6)

    ax.set_xlabel(r'SR/$d$')
    ax.set_ylabel('Avg. Downstream Accuracy')
    ax.set_xscale('log')
    add_subtle_grid(ax)

    save_fig(fig, 'fig08_correlation')


if __name__ == '__main__':
    main()
