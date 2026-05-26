# Visualization Upgrade Plan — Publication-Quality Figures

**Date**: 2026-05-25  
**Goal**: Elevate figure quality to top-tier AI venue standards (NeurIPS/ICML spotlight level)

---

## Design Philosophy

### Color Palette
- **Primary**: Low-saturation, high-contrast palette inspired by Nature/Science publications
- **Approach**: Muted blues, warm grays, desaturated reds — avoid fully saturated colors
- **Differentiation**: Use luminance + hue together (accessible for color-blind readers)
- **Background**: Pure white, minimal grid (alpha=0.15), no chart junk

### Typography
- Serif body text (Times/STIX), sans-serif labels where needed
- Font sizes: axis labels 8pt, tick labels 7pt, annotations 6-7pt
- LaTeX math rendering for all equations

### Layout Principles
- Single-column figures: 3.25" wide (NeurIPS)
- Double-column figures: 6.75" wide
- Consistent margins, aligned axes across multi-panel figures
- White space > decoration

---

## Figure Inventory (12 figures planned)

### Figure 1: Spectral Phase Portrait (NEW — Hero Figure)
**Type**: 2D trajectory plot  
**Data**: All 6 Pythia models + OLMo-2-13B  
**Axes**: x = SR/d, y = α  
**Visual**: Each model is a trajectory with time-arrows, colored by model size  
**Key insight**: All trajectories converge toward lower-left (compressed + structured), but large models stall  
**Source data**: `results/pythia_v2/*.jsonl`, `results/olmo2_v2/olmo2_13b.jsonl`

### Figure 2: SR/d Convergence (UPGRADED from current Fig 1)
**Type**: Multi-line time series  
**Data**: 6 Pythia models  
**Upgrade**: Add confidence band for final convergence target, add inset zoom on final 20%  
**Source data**: `results/pythia_v2/*.jsonl`

### Figure 3: α Dynamics with MLP/Attn Decomposition (UPGRADED from current Fig 2)
**Type**: Multi-panel (1×2 or 1×3)  
**Panel A**: Overall α for representative models (70M, 1B, 2.8B, 6.9B, 13B)  
**Panel B**: α_attn vs α_mlp for OLMo-2-13B (shaded reversal region)  
**Panel C** (optional): MLP/Attn gap evolution over time  
**Source data**: `results/pythia_v2/*.jsonl`, `results/olmo2_v2/olmo2_13b.jsonl`

### Figure 4: 3-Way Schedule Comparison (UPGRADED from current Fig 3)
**Type**: Multi-panel (1×3 or 2×2)  
**Panel A**: Training loss curves (3 schedules × 2 seeds, mean + CI band)  
**Panel B**: α dynamics during training  
**Panel C**: Downstream benchmark results (grouped bar chart with error bars)  
**Source data**: `results/real_3way/*.log`, `results/eval_410m/summary.json`

### Figure 5: SR/d vs Hidden Dimension (UPGRADED from current Fig 4)
**Type**: Scatter with fitted curve  
**Upgrade**: Better point labeling, architecture-grouped legend, add Mistral data point  
**Source data**: Hardcoded from measurements (all 13 models)

### Figure 6: Per-Layer Spectral Heatmap (NEW)
**Type**: 2D heatmap (layers × singular value rank)  
**Data**: Pythia-1B at step 0 vs step 143000  
**Visual**: Side-by-side heatmaps showing concentration of eigenvalue mass  
**Note**: May need to re-compute from checkpoints (load model, SVD each layer, store top-256 singular values)  
**Fallback**: Use existing concentration_top1/5/10 data to create a simplified version

### Figure 7: MLP vs Attention Gap (UPGRADED from current Fig 5)
**Type**: Grouped bar chart OR lollipop chart  
**Upgrade**: Add Mistral-7B, sort by gap size, add threshold annotations  
**Source data**: Hardcoded from measurements

### Figure 8: Structural Chinchilla / Phase Transition (UPGRADED from current Fig 6)
**Type**: Dual-axis or two-panel  
**Panel A**: α_final vs D/N (log-scale) with phase-transition sigmoid overlay  
**Panel B**: α_final vs N (showing the sharp transition at 1.7B)  
**Source data**: Hardcoded from measurements

### Figure 9: Downstream Performance Correlation (NEW)
**Type**: Scatter plot with regression line  
**Data**: SR/d vs average benchmark score (N=102 points: 6 models × 17 checkpoints)  
**Visual**: Points colored by model size, single regression line, annotate R² and Spearman r  
**Source data**: `results/pythia_v2/*.jsonl` + `results/pythia_benchmarks/*.json`

### Figure 10: Training Dynamics Dashboard (NEW — Conceptual)
**Type**: Annotated time series showing the monitoring framework  
**Data**: Single model (Pythia-1B or 410M α-guided run)  
**Visual**: Loss curve + α curve + SR/d curve on shared x-axis, with color-coded zones (HEALTHY/PLATEAU/REVERSAL) and trigger annotation  
**Source data**: `results/real_3way/alpha_s42.log`

### Figure 11: Universal Compression Law (NEW)
**Type**: Bar chart or scatter  
**Data**: Initial vs final SR/d for all models, showing ΔH₂ ≈ -2 nats universally  
**Visual**: Each model is a bar showing the compression amount, horizontal line at -2 nats  
**Source data**: All JSONL files (initial + final stable_rank_mean / d)

### Figure 12: Cross-Architecture Validation Grid (NEW)
**Type**: Multi-panel grid (4×2 or similar)  
**Data**: 4 architectures (GPT-NeoX, OLMo2, LLaMA, Mistral) × 2 metrics (SR/d trajectory, α trajectory)  
**Visual**: Show that the same patterns appear regardless of architecture  
**Source data**: All available JSONL files

---

## Data Requirements

### Available Locally (ready to use)
- [x] `results/pythia_v2/*.jsonl` — 6 Pythia models, 21 checkpoints each
- [x] `results/olmo2_v2/*.jsonl` — 4 OLMo-2 models (1B, 7B, 13B, 32B)
- [x] `results/amber_v2/*.jsonl` — Amber-7B
- [x] `results/k2_v2/*.jsonl` — K2-65B
- [x] `results/mistral_v2/*.jsonl` — Mistral-7B
- [x] `results/real_3way/*.log` — 3-way schedule training logs
- [x] `results/eval_410m/*.json` — Downstream benchmark results
- [x] `results/pythia_benchmarks/*.json` — 6 scales × 17 checkpoints × 87 tasks

### Needs Computation (can be derived)
- [ ] Per-layer singular value distributions (requires loading checkpoints → not available locally)
- [ ] Full eigenvalue spectra visualization (same)

### Fallback Strategy
For Figure 6 (per-layer heatmap): use the existing `concentration_top1/5/10` and `norm_entropy_mean` data to create a simplified but still informative visualization showing concentration patterns across training.

---

## Implementation Order

1. **Phase 1** (immediate, all data available):
   - Figure 1 (Phase Portrait) — hero figure, maximum impact
   - Figure 4 (3-Way upgraded) — includes new downstream data
   - Figure 9 (Correlation scatter) — strong quantitative claim
   - Figure 10 (Dashboard) — practical value showcase

2. **Phase 2** (requires minor data processing):
   - Figure 2 (SR/d upgraded)
   - Figure 3 (α with decomposition)
   - Figure 11 (Universal compression)
   - Figure 12 (Cross-architecture grid)

3. **Phase 3** (upgrades to existing):
   - Figure 5 (SR/d vs d upgraded)
   - Figure 7 (MLP/Attn gap upgraded)
   - Figure 8 (Structural Chinchilla upgraded)

4. **Phase 4** (if checkpoint data becomes available):
   - Figure 6 (Per-layer heatmap)

---

## Color Specification

```python
# Primary palette — muted, sophisticated
COLORS = {
    'blue_dark': '#2C3E50',      # Primary line color
    'blue_mid': '#5B7FA4',       # Secondary lines
    'blue_light': '#A8C5DA',     # Confidence bands
    'red_muted': '#C0392B',      # Accent/warning
    'red_light': '#E8A9A3',      # Light accent
    'green_muted': '#27AE60',    # Success/healthy
    'green_light': '#A9DFBF',    # Light success
    'orange_muted': '#E67E22',   # Caution
    'gray_dark': '#4A4A4A',      # Text
    'gray_mid': '#8E8E8E',       # Secondary text
    'gray_light': '#D4D4D4',     # Grid lines
    'purple_muted': '#8E44AD',   # Tertiary accent
    'gold': '#C19A00',           # Highlight
}

# Model size gradient (small → large)
MODEL_COLORS = [
    '#4A90D9',  # 70M (lightest blue)
    '#3D7CC4',  # 160M
    '#2F68AF',  # 410M
    '#22549A',  # 1B
    '#154085',  # 2.8B
    '#082C70',  # 6.9B (darkest blue)
]

# Architecture colors
ARCH_COLORS = {
    'GPT-NeoX': '#2C3E50',
    'OLMo2': '#E67E22',
    'LLaMA': '#27AE60',
    'Mistral': '#8E44AD',
}

# Schedule colors
SCHED_COLORS = {
    'cosine': '#8E8E8E',        # Gray (baseline)
    'wsd': '#2C3E50',           # Dark blue (established)
    'alpha_guided': '#C0392B',  # Muted red (ours, highlight)
}
```

---

## Quality Checklist (per figure)

- [ ] Font size readable at column width (no text < 6pt)
- [ ] Color-blind safe (check with simulator)
- [ ] No unnecessary gridlines or borders
- [ ] Axis labels with proper LaTeX math
- [ ] Legend placed to minimize data occlusion
- [ ] Consistent styling across all figures
- [ ] Saved as both PDF (vector) and PNG (300 DPI)
- [ ] Tight bounding box (minimal whitespace)
- [ ] No aliasing artifacts
- [ ] Annotations don't overlap data points
