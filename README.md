# Beyond Loss Curves: Spectral Monitoring for Language Model Pretraining

> Training a frontier language model costs $100-500M, yet practitioners monitor this investment through a single signal: **the training loss**. We show that spectral properties of weight matrices provide physically grounded monitoring signals that loss fundamentally cannot.

## Key Results

Measuring 12 models across 3 architectures (GPT-NeoX, LLaMA, OLMo2) spanning 70M--65B parameters:

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Universal compression law**: SR/d ≈ 0.040 + 0.61/√d | 12 models, 3 archs; two models with same d but different N achieve identical SR/d |
| 2 | **Performance predictor**: SR/d achieves r = -0.92 with downstream benchmarks | N=102 checkpoint–benchmark pairs, p < 10⁻⁴¹ |
| 3 | **α reversal = structural degradation**: invisible to loss | OLMo-2-13B shows Δα = +2.71; larger models are more fragile |
| 4 | **α-guided schedule**: matches hand-tuned WSD, outperforms cosine by -0.054 loss | Real data (FineWeb-Edu, 10B tokens), 2 seeds |
| 5 | **Training sufficiency audit**: SR/d diagnoses K2-65B as under-trained | Corroborated by benchmark data + authors' acknowledgment |

## Phase Transition at N ≈ 1.7B

Our most surprising finding: a sharp structural phase transition separates small and large models.

- **N ≤ 1B**: easily achieve heavy-tail structure (α < 3) regardless of training budget
- **N > 1.7B**: exhibit persistent structural immaturity (α > 5) even with hundreds of tokens/parameter

This implies frontier models may be systematically undertrained from a structural perspective.

## Repository Structure

```
paper/                           # LaTeX source + compiled PDF (main.pdf)
  sections/                      # abstract, introduction, framework, experiments, discussion, conclusion
  figures/                       # Publication-quality figures (6 PDFs)
  references.bib

scripts/
  thermo/                        # Measurement & training scripts
    measure_pythia_v2.py         # V2 spectral measurement (Pythia)
    measure_olmo2_v2.py          # V2 measurement (OLMo-2: 1B/7B/13B/32B)
    measure_k2_v2.py             # V2 measurement (K2-65B)
    measure_amber_v2.py          # V2 measurement (Amber-7B)
    train_real_data_3way.py      # 3-way schedule comparison (cosine/WSD/α-guided)
    prepare_data_fineweb.py      # Data tokenization for real-data experiment
  figures/
    plot_all.py                  # Generate all paper figures
  k8s/thermo/                    # Kubernetes job manifests

results/
  pythia_v2/                     # Pythia 70M-6.9B spectral measurements (6 JSONL)
  olmo2_v2/                      # OLMo-2 1B/7B/13B/32B measurements (4 JSONL)
  k2_v2/                         # K2-65B measurements (1 JSONL)
  amber_v2/                      # Amber-7B measurements (1 JSONL)
  real_3way/                     # 3-way schedule experiment logs (6 files)
  pythia_benchmarks/             # Downstream benchmark data (102 JSON)
  structural_chinchilla_refit.md # Scaling law refit analysis

docs/
  THEORY_V2.md                   # Complete theoretical framework + all experimental findings
  THEORY_UPGRADE.md              # Theory upgrade plan (analogy → first principles)
  EXPERIMENT_PRESCRIPTIVE.md     # Prescriptive experiment design + results
  EXPERIMENT_LOG.md              # Chronological experiment diary
  MONITORING_FRAMEWORK.md        # Practical monitoring guide
  MEASUREMENT_REVISION.md        # V1 → V2 metric revision history
```

## Models Measured

| Model | Architecture | d | Params | D/N | SR/d (final) | α (final) |
|-------|-------------|-----|--------|-----|-------------|-----------|
| Pythia-70M | GPT-NeoX | 512 | 70M | 4261 | 0.074 | 2.60 |
| Pythia-160M | GPT-NeoX | 768 | 162M | 1848 | 0.054 | 2.63 |
| Pythia-410M | GPT-NeoX | 1024 | 405M | 740 | 0.056 | 2.73 |
| Pythia-1B | GPT-NeoX | 2048 | 1.0B | 297 | 0.050 | 2.78 |
| Pythia-2.8B | GPT-NeoX | 2560 | 2.8B | 108 | 0.052 | 5.16 |
| Pythia-6.9B | GPT-NeoX | 4096 | 6.9B | 44 | 0.046 | 5.13 |
| Amber-7B | LLaMA | 4096 | 6.7B | 187 | 0.057 | 5.25 |
| K2-65B | LLaMA | 8192 | 65B | 21 | 0.036* | 5.09 |
| OLMo-2-1B | OLMo2 | 2048 | 1.0B | 4000 | 0.064 | 2.37 |
| OLMo-2-7B | OLMo2 | 4096 | 7.0B | 571 | 0.046 | 6.08 |
| OLMo-2-13B | OLMo2 | 5120 | 13B | 365 | 0.043 | 6.95 |
| OLMo-2-32B | OLMo2 | 5120 | 32B | 189 | 0.043 | 5.25 |

*K2-65B has not converged (D/N=21); SR/d reflects training insufficiency.

## Documentation Index

| Document | Purpose |
|----------|---------|
| [THEORY_V2.md](docs/THEORY_V2.md) | Master theory doc — all findings with timestamps and round indices |
| [THEORY_UPGRADE.md](docs/THEORY_UPGRADE.md) | Theory deepening plan (SR=exp(H₂), Langevin, Landau) |
| [EXPERIMENT_PRESCRIPTIVE.md](docs/EXPERIMENT_PRESCRIPTIVE.md) | Schedule experiments design + results |
| [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | Chronological experiment progress |
| [structural_chinchilla_refit.md](results/structural_chinchilla_refit.md) | Scaling law refit analysis (10 data points) |

## Citation

```bibtex
@inproceedings{anonymous2026beyond,
  title={Beyond Loss Curves: Thermodynamics of Pretraining},
  author={Anonymous},
  booktitle={NeurIPS 2026},
  year={2026}
}
```
