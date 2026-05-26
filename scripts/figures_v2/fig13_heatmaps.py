"""
Figure 13: Spectral Heatmaps (RLVR-inspired visual style)

Multiple heatmap-style visualizations showing spectral structure evolution:
(a) α evolution across models × training progress (2D heatmap)
(b) Concentration buildup across scales × progress
(c) MLP/Attn divergence heatmap

Data: results/pythia_v2/*.jsonl + results/olmo2_v2/*.jsonl
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# Custom colormaps (low-saturation, sophisticated)
CMAP_BLUE = LinearSegmentedColormap.from_list('custom_blue',
    ['#F7FBFF', '#C5DAE8', '#6BAED6', '#2171B5', '#08306B'])
CMAP_HEAT = LinearSegmentedColormap.from_list('custom_heat',
    ['#F7FCF5', '#C7E9C0', '#74C476', '#238B45', '#00441B'])
CMAP_DIVERGE = LinearSegmentedColormap.from_list('custom_div',
    ['#2166AC', '#92C5DE', '#F7F7F7', '#F4A582', '#B2182B'])
CMAP_VIRIDIS_MUTED = LinearSegmentedColormap.from_list('muted_viridis',
    ['#440154', '#3B528B', '#21918C', '#5DC863', '#FDE725'])


def main():
    fig = plt.figure(figsize=(TEXT_W, 5.5))

    # Layout: 2 rows, top row = 1 wide heatmap, bottom row = 2 panels
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Full width
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # =========================================================================
    # Panel A: α evolution heatmap (models × training progress)
    # =========================================================================
    all_models = [
        ('pythia_v2/pythia_70m.jsonl', 'Pythia-70M', 512),
        ('pythia_v2/pythia_160m.jsonl', 'Pythia-160M', 768),
        ('pythia_v2/pythia_410m.jsonl', 'Pythia-410M', 1024),
        ('pythia_v2/pythia_1b.jsonl', 'Pythia-1B', 2048),
        ('pythia_v2/pythia_2.8b.jsonl', 'Pythia-2.8B', 2560),
        ('pythia_v2/pythia_6.9b.jsonl', 'Pythia-6.9B', 4096),
        ('olmo2_v2/olmo2_1b.jsonl', 'OLMo-2-1B', 2048),
        ('olmo2_v2/olmo2_7b.jsonl', 'OLMo-2-7B', 4096),
        ('olmo2_v2/olmo2_13b.jsonl', 'OLMo-2-13B', 5120),
        ('olmo2_v2/olmo2_32b.jsonl', 'OLMo-2-32B', 5120),
    ]

    # Interpolate all to common x-axis (0-100% in 20 bins)
    n_bins = 20
    progress_bins = np.linspace(0, 100, n_bins)
    alpha_matrix = np.full((len(all_models), n_bins), np.nan)
    model_labels = []

    for i, (path_str, label, d) in enumerate(all_models):
        path = RESULTS / path_str
        if not path.exists():
            model_labels.append(label)
            continue
        records = load_jsonl(path)
        if len(records) < 2:
            model_labels.append(label)
            continue
        max_step = records[-1]['step']
        if max_step == 0:
            model_labels.append(label)
            continue

        frac = np.array([r['step'] / max_step * 100 for r in records])
        alpha = np.array([r['alpha_mean'] for r in records])
        model_labels.append(label)

        # Interpolate to bins
        for j, bin_val in enumerate(progress_bins):
            # Find nearest
            idx = np.argmin(np.abs(frac - bin_val))
            alpha_matrix[i, j] = alpha[idx]

    # Clip for better visualization
    alpha_clipped = np.clip(alpha_matrix, 2, 12)

    im = ax1.imshow(alpha_clipped, aspect='auto', cmap=CMAP_VIRIDIS_MUTED,
                     interpolation='bilinear', vmin=2, vmax=10,
                     extent=[0, 100, len(all_models) - 0.5, -0.5])

    ax1.set_yticks(range(len(model_labels)))
    ax1.set_yticklabels(model_labels, fontsize=6.5)
    ax1.set_xlabel('Training Progress (%)')
    ax1.set_title(r'(a) $\alpha$ Evolution Across Models (lower = more structured)',
                  fontsize=8.5, pad=6)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.02)
    cbar.set_label(r'$\alpha$', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Add dividing line between Pythia and OLMo
    ax1.axhline(5.5, color='white', linewidth=1.5, ls='-')
    ax1.text(-3, 2.5, 'Pythia', fontsize=6, color=C['blue'], rotation=90,
             va='center', ha='right', fontweight='medium')
    ax1.text(-3, 7.5, 'OLMo-2', fontsize=6, color=C['orange'], rotation=90,
             va='center', ha='right', fontweight='medium')

    # =========================================================================
    # Panel B: Concentration buildup heatmap
    # =========================================================================
    conc_matrix = np.full((6, n_bins), np.nan)  # Pythia only
    pythia_labels = []

    for i, (fname, label, d) in enumerate(PYTHIA_MODELS):
        path = RESULTS / 'pythia_v2' / f'{fname}.jsonl'
        if not path.exists():
            pythia_labels.append(label)
            continue
        records = load_jsonl(path)
        max_step = records[-1]['step'] if records else 1
        frac = np.array([r['step'] / max_step * 100 for r in records])
        conc = np.array([r.get('concentration_top10', 0) for r in records])
        pythia_labels.append(label)

        for j, bin_val in enumerate(progress_bins):
            idx = np.argmin(np.abs(frac - bin_val))
            conc_matrix[i, j] = conc[idx]

    im2 = ax2.imshow(conc_matrix, aspect='auto', cmap=CMAP_HEAT,
                      interpolation='bilinear', vmin=0, vmax=0.20,
                      extent=[0, 100, 5.5, -0.5])

    ax2.set_yticks(range(len(pythia_labels)))
    ax2.set_yticklabels(pythia_labels, fontsize=6.5)
    ax2.set_xlabel('Training Progress (%)')
    ax2.set_title('(b) Top-10 Eigenvalue Concentration', fontsize=8.5, pad=6)

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
    cbar2.set_label('Concentration', fontsize=6.5)
    cbar2.ax.tick_params(labelsize=6)

    # =========================================================================
    # Panel C: SR/d convergence heatmap (Pythia only — matches panel B)
    # =========================================================================
    srd_matrix = np.full((6, n_bins), np.nan)

    for i, (fname, label, d) in enumerate(PYTHIA_MODELS):
        path = RESULTS / 'pythia_v2' / f'{fname}.jsonl'
        if not path.exists():
            continue
        records = load_jsonl(path)
        max_step = records[-1]['step'] if records else 1
        frac = np.array([r['step'] / max_step * 100 for r in records])
        sr_d = np.array([r['stable_rank_mean'] / d for r in records])

        for j, bin_val in enumerate(progress_bins):
            idx = np.argmin(np.abs(frac - bin_val))
            srd_matrix[i, j] = sr_d[idx]

    # Clip for visualization
    srd_clipped = np.clip(srd_matrix, 0.03, 0.45)

    im3 = ax3.imshow(srd_clipped, aspect='auto', cmap=CMAP_BLUE,
                      interpolation='bilinear', vmin=0.03, vmax=0.45,
                      extent=[0, 100, 5.5, -0.5])

    ax3.set_yticks(range(len(pythia_labels)))
    ax3.set_yticklabels(pythia_labels, fontsize=6.5)
    ax3.set_xlabel('Training Progress (%)')
    ax3.set_title(r'(c) SR/$d$ Compression (darker = more compressed)',
                  fontsize=8.5, pad=6)

    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)
    cbar3.set_label(r'SR/$d$', fontsize=6.5)
    cbar3.ax.tick_params(labelsize=6)

    save_fig(fig, 'fig13_heatmaps')


if __name__ == '__main__':
    main()
