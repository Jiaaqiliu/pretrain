"""
Shared style configuration and utilities for publication figures.

Usage:
    from style import *
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    ...
    save_fig(fig, 'my_figure')
"""

import json
import re
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

# =============================================================================
# Global Style — Low-saturation, sophisticated
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX', 'STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#cccccc',
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 1.3,
    'lines.markersize': 5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# =============================================================================
# Color System — Muted, Nature/Science style
# =============================================================================
C = {
    'navy': '#1B2838',
    'blue': '#3A6B8C',
    'blue_light': '#7BA7C2',
    'blue_pale': '#C5DAE8',
    'red': '#9B2335',
    'red_light': '#C7727F',
    'red_pale': '#E8BFC5',
    'green': '#2D6A4F',
    'green_light': '#74B49B',
    'orange': '#B85C38',
    'orange_light': '#D4956A',
    'purple': '#5B3A6B',
    'purple_light': '#9B7DAD',
    'gold': '#8B7021',
    'gray': '#5A5A5A',
    'gray_light': '#A0A0A0',
    'gray_pale': '#E0E0E0',
}

# Model size gradient (6 levels: 70M → 6.9B)
MODEL_GRADIENT = ['#6FA8C5', '#4C8DAF', '#357399', '#265A82', '#18416B', '#0B2D55']

# Architecture styles
ARCH_STYLE = {
    'GPT-NeoX': {'color': C['blue'], 'marker': 'o', 'label': 'GPT-NeoX (Pythia)'},
    'OLMo2': {'color': C['orange'], 'marker': '^', 'label': 'OLMo-2'},
    'LLaMA': {'color': C['green'], 'marker': 's', 'label': 'LLaMA (Amber/K2)'},
    'Mistral': {'color': C['purple'], 'marker': 'D', 'label': 'Mistral'},
}

# Schedule colors
SCHED = {
    'cosine': {'color': C['gray'], 'ls': '-', 'label': 'Cosine'},
    'wsd': {'color': C['blue'], 'ls': '-', 'label': 'WSD'},
    'alpha': {'color': C['red'], 'ls': '-', 'label': r'$\alpha$-Guided (Ours)'},
}

# OLMo color gradient
OLMO_COLORS = ['#D4956A', '#B85C38', '#8B4226', '#5C2D14']

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_ROOT / 'results'
FIG_OUT = PROJECT_ROOT / 'paper' / 'figures_v2'
FIG_OUT.mkdir(parents=True, exist_ok=True)

# NeurIPS sizing
COL_W = 3.25   # single column width (inches)
TEXT_W = 6.75  # full text width

# =============================================================================
# Data Loaders
# =============================================================================
def load_jsonl(path):
    """Load spectral measurement JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if 'alpha_mean' in r and 'stable_rank_mean' in r:
                records.append(r)
    records.sort(key=lambda r: r['step'])
    return records


def load_log(path):
    """Load training log file with loss and spectral measurements."""
    records = []
    with open(path) as f:
        for line in f:
            m = re.search(r'step (\d+)/\d+: loss=([\d.]+), lr=([\d.e+-]+)', line)
            if m:
                records.append({
                    'step': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'lr': float(m.group(3))
                })
            m2 = re.search(r'\[SPECTRAL\] step (\d+): .=([\d.]+), SR/d=([\d.]+)', line)
            if m2:
                step = int(m2.group(1))
                found = False
                for r in records:
                    if r['step'] == step:
                        r['alpha'] = float(m2.group(2))
                        r['sr_d'] = float(m2.group(3))
                        found = True
                        break
                if not found:
                    records.append({
                        'step': step,
                        'alpha': float(m2.group(2)),
                        'sr_d': float(m2.group(3))
                    })
    return records


# =============================================================================
# Utility Functions
# =============================================================================
def add_subtle_grid(ax, axis='both', alpha=0.12):
    """Add barely-visible grid lines."""
    ax.grid(True, axis=axis, alpha=alpha, linewidth=0.3, color='#888888')


def save_fig(fig, name):
    """Save figure as PDF + PNG."""
    fig.savefig(FIG_OUT / f'{name}.pdf')
    fig.savefig(FIG_OUT / f'{name}.png')
    plt.close(fig)
    print(f'  ✓ Saved: {name}.pdf + {name}.png')


# Pythia model definitions
PYTHIA_MODELS = [
    ('pythia_70m', '70M', 512),
    ('pythia_160m', '160M', 768),
    ('pythia_410m', '410M', 1024),
    ('pythia_1b', '1B', 2048),
    ('pythia_2.8b', '2.8B', 2560),
    ('pythia_6.9b', '6.9B', 4096),
]

OLMO_MODELS = [
    ('olmo2_1b', '1B', 2048),
    ('olmo2_7b', '7B', 4096),
    ('olmo2_13b', '13B', 5120),
    ('olmo2_32b', '32B', 5120),
]
