# Experiment Design: Pretraining Data Mixtures as a Search Problem

## Research Thesis

> Automated search over pretraining data mixtures — including domain ratios, filter
> aggressiveness, and training curriculum — discovers recipes that transfer from proxy
> models (190M) to target scale (3B) and produce measurable gains on downstream
> reasoning tasks (Kaggle NeMo Reasoning).

## Core Research Questions

**RQ1** (Primary): Can MCGS find data mixtures at 190M scale that outperform
expert-designed baselines (OLMo-2, Llama-3) when applied at 3B scale?

**RQ2** (Bitter Lesson): Does the optimal filter aggressiveness depend on model size?
Specifically, does reasoning performance require more aggressive filtering than general
language modeling, or does the Bitter Lesson still hold?

**RQ3** (Curriculum): Does a learned data curriculum (phased transitions in domain
ratios) outperform any static mixture? At what training stage do curriculum shifts
provide the most gain?

**RQ4** (Efficiency): How many proxy trials are needed to find a recipe that improves
over the baseline? Is LLM-guided mutation more sample-efficient than random search?

---

## Experimental Protocol

### Baselines (from actual technical reports)

| Name | Web | Code | Math | Books | Academic | Source |
|------|-----|------|------|-------|----------|--------|
| OLMo-2 Stage1 | 95.1% | 2.1% | 0.6% | 0.1% | 2.0% | Allen AI tech report (Jan 2025) |
| OLMo-2 Midtrain | 47.2% | 2.5% | 19.6% | 7.1% | 5.2% | Allen AI (50B annealing mix) |
| Llama-3 | 50% | 17% | 25% | 4% | 4% | Meta (Jul 2024) |
| DeepSeek-v3 (est) | 45% | 25% | 20% | 5% | 5% | DeepSeek (Dec 2024) |
| DoReMi | 60.6% | 1.8% | 0.4% | 2.2% | 3.5% | NeurIPS 2023 |
| Uniform | 20% | 20% | 20% | 20% | 20% | Control |
| Reasoning-heavy | 30% | 30% | 25% | 5% | 10% | Our hypothesis |

**Key observation**: OLMo-2's Stage 1 is 95% web, but their midtraining mix shifts dramatically
toward math (20%). Llama-3's final mix has 25% math+reasoning. This suggests the optimal recipe
is NOT a single static mix — it's a staged curriculum. Our MCGS should discover this automatically.

**DCLM transfer validation**: Recipe ranking correlation from proxy to target:
- 400M → 7B: r=0.838
- 1B → 7B: r=0.956
- 3B → 7B: r=0.982

This strongly supports our proxy approach (190M → 3B → 7B).

### Search Axes (Controlled Variables)

**Experiment 1: Static Mix Search**
- Vary: domain ratios (5-dim simplex)
- Fixed: LR=3e-4, warmup=500, cosine, batch=64, filter=default

**Experiment 2: Filter Aggressiveness**
- Vary: per-domain retention rate [0.05, 1.0]
- Fixed: OLMo-2 mix ratios, same training config
- Key comparison: Does less filtering help more at 3B than at 190M?

**Experiment 3: Curriculum Search**
- Vary: 2-phase and 3-phase curricula, transition points
- Fixed: Same total compute budget
- Key question: When should you shift toward reasoning-heavy data?

**Experiment 4: Joint Search (Full MCGS)**
- Vary: mix + filter + curriculum simultaneously
- Compare: joint search vs optimizing each axis independently

### Training Configuration (Held Constant)

| Parameter | 190M (proxy) | 3B (target) |
|-----------|-------------|-------------|
| Model | olmo2_190M | olmo2_3B |
| Steps per trial | 5,000 | 10,000 |
| Sequence length | 4,096 | 4,096 |
| Global batch (tokens) | 262,144 | 1,048,576 |
| Learning rate | 3e-4 | 3e-4 |
| Warmup | 500 steps | 1,000 steps |
| Schedule | Cosine | Cosine |
| Optimizer | AdamW (β=0.9,0.95) | AdamW (β=0.9,0.95) |
| Precision | bf16 | bf16 |
| Parallelism | FSDP (8 GPU) | FSDP (16 GPU) |

### Evaluation Metrics

**Primary (optimized by MCGS):**
- C4 validation loss (general LM quality)

**Secondary (measured but not optimized in proxy):**
- Per-domain validation loss: web, code, math
- ARC-Easy accuracy (reasoning)
- PIQA accuracy (commonsense)
- HellaSwag accuracy (commonsense)

**Downstream (Phase 3 only):**
- Full benchmark suite: MMLU, GSM8K, HumanEval, ARC, PIQA, HellaSwag, WinoGrande
- Kaggle NeMo Reasoning score (after standard LoRA post-training)

---

## Phase Plan

### Phase 1: Proxy Search (Weeks 1-3)

**Experiment 1a: Manual Baselines (Week 1)**
- Train 5 baseline mixtures at 190M for 5000 steps
- Verify eval pipeline works, establish variance estimates
- Compute: 5 × 2 GPU-hours = 10 GPU-hours

**Experiment 1b: MCGS Static Mix Search (Weeks 1-2)**
- 40 MCGS cycles, static mixtures only
- Starting from OLMo-2 mixture
- Compute: 40 × 2 GPU-hours = 80 GPU-hours

**Experiment 1c: MCGS Full Search (Weeks 2-3)**
- 40 more cycles, adding curriculum + filter dimensions
- Starting from best of 1b
- Compute: 40 × 2 GPU-hours = 80 GPU-hours

**Deliverable**: Top-5 static mixes + top-3 curricula + filter analysis

### Phase 2: Transfer Verification (Weeks 4-6)

**Experiment 2a: Top Recipes at 3B (Week 4-5)**
- Run top-3 discovered recipes + 2 baselines at 3B, 10K steps
- Compute: 5 × 100 GPU-hours = 500 GPU-hours

**Experiment 2b: Bitter Lesson at 3B (Week 5-6)**
- 3 filter levels × 2 mixes at 3B, 5K steps each
- Test if more data / less filter helps more at 3B
- Compute: 6 × 50 GPU-hours = 300 GPU-hours

**Deliverable**: Transfer correlation (Spearman ρ), Bitter Lesson analysis

### Phase 3: Full Training + Kaggle (Weeks 7-9)

**Experiment 3a: Full 60B Token Training (Week 7-8)**
- Best recipe: 3B, 60K steps (full Chinchilla-optimal)
- OLMo-2 baseline: same setup with official mix
- Compute: 2 × 1000 GPU-hours = 2000 GPU-hours

**Experiment 3b: Post-training + Kaggle (Week 8-9)**
- Same LoRA SFT pipeline applied to both checkpoints
- Submit to Kaggle NeMo Reasoning
- Compute: 2 × 50 GPU-hours = 100 GPU-hours

**Deliverable**: Main results table, Kaggle score, paper draft

### Phase 4: Ablations + Paper (Weeks 10-12)

**Experiment 4a: Search Efficiency Ablation**
- Compare: MCGS+LLM vs MCGS+random vs Bayesian vs Random
- Same total budget (40 trials each), measure best-found loss
- Compute: 3 × 80 GPU-hours = 240 GPU-hours

**Experiment 4b: Which Dimensions Matter?**
- SHAP-like analysis: contribution of each search dimension
- Freeze all but one dimension, measure marginal improvement

**Deliverable**: Complete paper with all figures and tables

---

## Total Compute Budget

| Phase | GPU-hours | Cost ($3/hr) |
|-------|-----------|--------------|
| Phase 1 | 170 | $510 |
| Phase 2 | 800 | $2,400 |
| Phase 3 | 2,100 | $6,300 |
| Phase 4 | 240 | $720 |
| **Total** | **3,310** | **$9,930** |

---

## Key Hypotheses and Expected Outcomes

### H1: Mix Search Finds Non-Obvious Recipes
**Prediction**: MCGS will discover mixes with higher code+math ratios than
any published recipe (possibly >40% combined), especially for reasoning.
**Justification**: DoReMi showed 2x more code than Pile's original ratio helps.

### H2: Curriculum Beats Static
**Prediction**: A 2-phase curriculum (web-heavy → reasoning-heavy) will beat
the best static mix by 2-5% in downstream reasoning.
**Justification**: Models learn structure from web text first, then specialize.

### H3: Bitter Lesson is Weaker for Reasoning
**Prediction**: Unlike general LM (where unfiltered wins at scale), for
reasoning tasks the crossover point is much later or doesn't exist.
**Justification**: Reasoning requires coherent multi-step signal that noise
disrupts more than simple next-token prediction.

### H4: Proxy Transfer Works
**Prediction**: Spearman correlation between 190M and 3B recipe rankings
will be ρ > 0.7 for data mix, but ρ < 0.5 for filter aggressiveness.
**Justification**: Mix ratios are scale-invariant; filter optimal point shifts.

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Recipes don't transfer 190M→3B | 20% | High | Use muTransfer-style correction; report as finding |
| All mixes perform similarly | 15% | Medium | Expand to curriculum/filter axes; smaller differences still publishable |
| Compute budget exceeded | 30% | Medium | Reduce to 30 cycles; use early stopping |
| Kaggle score not competitive | 25% | Low | Paper claim doesn't require winning; show improvement over baseline |
| Bitter Lesson holds for reasoning too | 40% | Low | This is also a valid finding; refine the paper framing |

---

## Reproduction Instructions

```bash
# 1. Setup environment
cd /path/to/A-EVOLVE-V2
pip install -e .[all]
pip install -e olmo-core/.[all]

# 2. Prepare data (requires HuggingFace access + FSx storage)
python scripts/prepare_data_3b.py

# 3. Run proxy search (requires 8x H200)
python experiments/autopretrain/run_proxy_search.py \
    --cycles 50 \
    --model olmo2_190M \
    --steps-per-trial 5000 \
    --output-dir results/phase1_proxy

# 4. Analyze results
python experiments/autopretrain/analyze_search.py results/phase1_proxy

# 5. Transfer to 3B (requires 16x H200)
python experiments/autopretrain/run_transfer_verification.py \
    --recipes results/phase1_proxy/final_results.json \
    --model olmo2_3B \
    --steps 10000
```
