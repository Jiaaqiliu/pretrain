"""
Master script: Generate all publication figures.

Run: python scripts/figures_v2/generate_all.py
Or run individual figures: python scripts/figures_v2/fig01_phase_portrait.py
"""
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FIGURES = [
    'fig01_phase_portrait',
    'fig02_srd_convergence',
    'fig03_alpha_dynamics',
    'fig04_3way_comparison',
    'fig05_srd_vs_d',
    'fig06_mlp_attn_gap',
    'fig07_phase_transition',
    'fig08_correlation',
    'fig09_dashboard',
    'fig10_compression',
    'fig11_cross_arch',
    'fig12_concentration',
    'fig13_heatmaps',
    'fig14_weight_structure',
]

if __name__ == '__main__':
    print('=' * 60)
    print('Generating all publication figures (v2)')
    print('Style: Low-saturation, NeurIPS-compatible, 300 DPI')
    print('=' * 60)
    print()

    success = 0
    failed = []

    for fig_name in FIGURES:
        try:
            mod = importlib.import_module(fig_name)
            mod.main()
            success += 1
        except Exception as e:
            print(f'  ✗ {fig_name}: {e}')
            failed.append((fig_name, str(e)))

    print()
    print('=' * 60)
    print(f'Results: {success}/{len(FIGURES)} figures generated successfully')
    if failed:
        print(f'Failed ({len(failed)}):')
        for name, err in failed:
            print(f'  - {name}: {err}')
    print(f'Output: paper/figures_v2/')
    print('=' * 60)
