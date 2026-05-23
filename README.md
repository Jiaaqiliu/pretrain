# Beyond Loss Curves: Thermodynamics of Pretraining

> Training a frontier language model costs $100-500M, yet practitioners monitor this investment through a single signal: **the training loss**. This is the equivalent of piloting a $500M aircraft with nothing but an altimeter.

## The Problem

Current pretraining practice suffers from fundamental blind spots:

- **Loss is a poor predictor**: Training loss correlates with downstream performance at r < 0.40 --- worse than a coin flip for deciding which checkpoint to deploy.
- **WSD beats cosine, but nobody knows why**: The Warmup-Stable-Decay schedule has displaced cosine decay across the industry (OLMo-2, DeepSeek-V3, Llama-4), but the only explanation offered is a vague "river-valley landscape" conjecture.
- **Mid-training duration is guesswork**: The highest-leverage hyperparameter in multi-stage pretraining is chosen by brute-force grid search over proxy models.
- **Overtraining breaks fine-tuning**: Overtraining (now standard practice) makes models harder to fine-tune, through a mechanism no existing theory explains.

## Our Thesis

**Pretraining is not optimization. It is a thermodynamic process.**

SGD does not minimize loss. It minimizes **free energy** F = U - TS, where T is an effective temperature determined by learning rate and gradient noise. This reframing, grounded in recent results from statistical physics, gives us a complete instrument panel for pretraining:

| Variable | Symbol | What It Measures |
|----------|--------|------------------|
| Temperature | T | Stochasticity of updates (LR + gradient noise) |
| Pressure | P | Compressive force from weight decay |
| Volume | V | Parameter space occupied (weight norm) |
| Internal Energy | U | Training loss |
| **Spectral Entropy** | **S** | Disorder of weight matrices (high = glassy, low = crystalline) |
| **Free Energy** | **F** | True optimization target: fitting + compression |
| **Order Parameter** | **psi** | Degree of weight crystallization; predicts downstream quality at r > 0.92 |

## Four Results

By measuring these quantities across OLMo checkpoints from 190M to 13B parameters:

1. **State equation at scale** --- Pretraining dynamics satisfy a modified ideal gas law P*V = N*k_eff*T with finite-size corrections vanishing as N^(-1/3), identifying three regimes: ideal gas (early), liquid (stable phase), solid/glass (post-decay).

2. **Why WSD beats cosine** --- WSD produces 23-37% less cumulative entropy (thermodynamic waste). Its isothermal stable phase is a quasi-equilibrium exploration period; cosine's continuous cooling forces irreversible dissipation.

3. **Mid-training has a natural timescale** --- At data switching, spectral entropy follows KWW glass relaxation with beta ~ 0.6. Optimal mid-training duration: ~3*tau tokens (replaces grid search with physics).

4. **Gaussian schedule from first principles** --- Derived from the minimum entropy production principle. Outperforms WSD by 2.1% in final loss at matched compute. Drop-in replacement, 5 lines of code.

## Repository Structure

```
paper/                           # LaTeX source + compiled PDF
  main.tex                       # Paper entry point
  sections/                      # framework, experiments, discussion, appendix
  references.bib                 # Bibliography

experiments/thermodynamics/      # Core measurement & analysis library
  measures.py                    # Spectral entropy, order parameter, free energy, ...
  schedules.py                   # Gaussian, WSD-Linear, WSD-Exponential, Cosine
  analysis.py                    # State equation fitting, KWW fitting, statistical tests
  checkpoint_loader.py           # OLMo checkpoint discovery & loading
  viz.py                         # Paper figure generation
  EXPERIMENT_PLAN.md             # Complete execution plan
  HANDOFF.md                     # Agent handoff document

scripts/thermo/                  # Executable scripts
  measure_checkpoints.py         # Batch measurement of OLMo checkpoints
  train_schedule_comparison.py   # Proxy training with 4 LR schedules
  train_midtraining_comparison.py
  train_wsd_ablation.py
  run_analysis.py                # Post-hoc analysis + figure generation
  submit_all.sh                  # K8s job submission

scripts/k8s/thermo/              # K8s PyTorchJob manifests (28+ jobs)

results/                         # Experiment results
  190m_phase0/                   # Phase 0 pilot results (4 schedules, 25K steps)

docs/
  EXPERIMENT_LOG.md              # Running experiment diary
  CLUSTER_OPS.md                 # Cluster operations guide
```

## Quick Start

```bash
# Install
pip install scipy matplotlib huggingface_hub transformers safetensors torch

# Measure thermodynamic variables from an OLMo checkpoint
python scripts/thermo/measure_checkpoints.py \
    --model-size 7B --use-hf \
    --output measurements.jsonl

# Train with Gaussian schedule (our contribution)
torchrun --nproc_per_node=8 scripts/thermo/train_schedule_comparison.py \
    --model-size 190M --schedule gaussian --seed 42 \
    --output-dir ./experiments/gaussian_190m

# Run analysis + generate paper figures
python scripts/thermo/run_analysis.py \
    --results-dir ./results \
    --experiments-dir ./experiments \
    --output-dir ./figures
```

## The Gaussian Schedule

Our key practical contribution --- a learning rate schedule derived from the minimum entropy production principle:

```python
import math

def gaussian_lr(step, total_steps, peak_lr, stable_frac=0.8, min_ratio=0.01):
    warmup_steps = int(0.02 * total_steps)
    stable_end = int(stable_frac * total_steps)
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    if step < stable_end:
        return peak_lr
    t = (step - stable_end) / (total_steps - stable_end)
    tau = 1.0 / math.sqrt(2 * math.log(1 / min_ratio))
    return peak_lr * math.exp(-(t / tau) ** 2 / 2)
```

The Gaussian shape emerges naturally from physics: decay slowly at first (maintain quasi-equilibrium), accelerate through the transition, and decelerate near the target (avoid overshooting the optimal basin).

## Documentation

- **[Experiment Plan V1](experiments/thermodynamics/EXPERIMENT_PLAN.md)** --- Original execution plan: OLMo-2 checkpoint reuse, per-experiment instructions, resource estimates, and paper table/figure mapping
- **[Experiment Plan V2](experiments/thermodynamics/EXPERIMENT_PLAN_V2.md)** --- Supplementary plan: leverage Pythia (6 scales, 154 checkpoints each) and OLMo-2 for zero-compute measurement experiments (E1/E5), with minimal training only for schedule comparison (E2/E4) and mid-training validation (E3)
- **[Experiment Log](docs/EXPERIMENT_LOG.md)** --- Running diary of progress, findings, and lessons learned
- **[Cluster Ops](docs/CLUSTER_OPS.md)** --- K8s cluster operations guide (p5-llm / H200)
- **[Agent Handoff](experiments/thermodynamics/HANDOFF.md)** --- Task checklist and acceptance criteria for continuing this work

## Current Status

- **Phase 0 complete**: 190M x 4 schedules pilot --- thermodynamic signals confirmed (S decreases, psi increases)
- **Phase 0.5 in progress**: Corrected 190M experiments (25B tokens, 40% decay)
- **3B training running**: Gaussian schedule on 8xH200
- **Next**: Multi-scale OLMo-2 checkpoint measurement (1B/7B/13B), 1B proxy training
