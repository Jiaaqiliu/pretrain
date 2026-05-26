"""Fig 15: Schedule advantage amplifies with scale (410M vs 1B)."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.15,
})

# Data
scales = ['410M', '1B']
cosine = [0.4594, 0.4859]
wsd = [0.4672, 0.4935]
alpha_guided = [0.4684, 0.4984]

x = np.arange(len(scales))
width = 0.22

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8), gridspec_kw={'width_ratios': [2, 1.2]})

# Panel A: Grouped bar chart
bars1 = ax1.bar(x - width, cosine, width, label='Cosine', color='#8E8E8E', edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x, wsd, width, label='WSD', color='#2C3E50', edgecolor='white', linewidth=0.5)
bars3 = ax1.bar(x + width, alpha_guided, width, label=r'$\alpha$-Guided', color='#C0392B', edgecolor='white', linewidth=0.5)

ax1.set_xlabel('Model Scale')
ax1.set_ylabel('Avg. Downstream Accuracy (5 tasks)')
ax1.set_xticks(x)
ax1.set_xticklabels(scales)
ax1.set_ylim(0.44, 0.51)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.set_title('(a) Downstream Performance by Scale', fontsize=9, pad=8)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=6.5)

# Panel B: Advantage amplification
alpha_vs_cosine = [(a - c) / c * 100 for a, c in zip(alpha_guided, cosine)]
alpha_vs_wsd = [(a - w) / w * 100 for a, w in zip(alpha_guided, wsd)]

x2 = np.arange(len(scales))
width2 = 0.3

bars_ac = ax2.bar(x2 - width2/2, alpha_vs_cosine, width2, label=r'$\alpha$-Guided vs Cosine',
                  color='#C0392B', alpha=0.8, edgecolor='white', linewidth=0.5)
bars_aw = ax2.bar(x2 + width2/2, alpha_vs_wsd, width2, label=r'$\alpha$-Guided vs WSD',
                  color='#E67E22', alpha=0.8, edgecolor='white', linewidth=0.5)

ax2.set_xlabel('Model Scale')
ax2.set_ylabel('Relative Improvement (%)')
ax2.set_xticks(x2)
ax2.set_xticklabels(scales)
ax2.set_ylim(0, 3.5)
ax2.legend(loc='upper left', framealpha=0.9, fontsize=7)
ax2.set_title(r'(b) $\alpha$-Guided Advantage Amplifies', fontsize=9, pad=8)

# Add value labels
for bars in [bars_ac, bars_aw]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'+{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7)

plt.tight_layout()

outdir = Path('/Users/itsjiaqi/Projects/pretrain-review/paper/figures_v2')
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / 'fig15_scale_comparison.pdf', bbox_inches='tight')
plt.savefig(outdir / 'fig15_scale_comparison.png', bbox_inches='tight')
print(f"Saved to {outdir / 'fig15_scale_comparison.pdf'}")
