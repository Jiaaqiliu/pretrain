# AutoPretrain Research Plan

## Overview

Two converging research directions that share infrastructure, code, and experimental pipeline.
Both are viable papers; the choice depends on compute budget and timeline.

---

## Related Work: "A Bitter Lesson for Data Filtering" (Mohri, Duchi, Hashimoto — Stanford, May 2025)

### Paper Summary

**Core claim**: With enough compute, the best data filter is NO data filter. Unfiltered Common Crawl
eventually outperforms all filtered variants (including DCLM-Baseline which retains only 2.1% of data)
when models are sufficiently large and trained long enough.

**Key findings**:
1. The crossover point where unfiltered > filtered is **predictable via scaling laws**
2. Large models (1B+) are remarkably robust to noise injection (even shuffled-word documents)
3. Current aggressive filtering (DCLM keeps ~1%) creates a token starvation problem at Chinchilla-optimal budgets
4. Theoretical explanation: large models have sufficient rank to compartmentalize mixed-quality data

**Their prediction**: At ~10^30 FLOPs (~2030 compute levels), unfiltered CC provably wins.

### Implications for Our Work

This paper is **extremely synergistic** with our research for three reasons:

1. **It validates the search framing**: If optimal filtering depends on compute budget, model size,
   and training duration, then the "right" data recipe is NOT a fixed answer — it's a function of
   your constraint envelope. This is EXACTLY what MCGS searches over.

2. **It opens a new search dimension**: Instead of "which filter to apply", the question becomes
   "how aggressively to filter given my compute budget". Our MCGS can search along this
   filter-aggressiveness axis as a continuous parameter.

3. **It gives us a strong baseline to beat**: Their scaling law predicts the crossover. If our
   MCGS-discovered recipe can beat both "fully filtered" and "fully unfiltered" regimes (by finding
   the Goldilocks zone), that's a novel contribution on top of their work.

**Critical insight for our paper**: The Bitter Lesson paper shows filtering is a compute-constrained
strategy. Our counter-argument: **intelligent data scheduling** (not just filtering) can beat both
extremes. The MCGS doesn't just decide "keep or discard" — it decides "when to show what data, in
what proportion, at what stage of training."

---

## Option A: "AutoPretrain: Scaling LLM-Guided Training Recipe Search to 30B Parameters"

### Positioning

- **Venue**: NeurIPS / ICML / ICLR (top venue)
- **Audience**: Pretraining researchers at Meta FAIR, OpenAI, DeepMind, Karpathy
- **Comparison papers**: Chinchilla, muTransfer, DoReMi, D4, Scaling Data-Constrained Language Models
- **Novelty**: First to demonstrate automated recipe search that transfers across 100x model scale

### Core Claims

**Claim 1**: LLM-guided MCGS on proxy models (190M-3B) discovers pretraining recipes that transfer
to 30B scale and outperform human-designed baselines (OLMo-2 official recipe).

**Claim 2**: The discovered recipes reveal non-obvious training dynamics (e.g., data curriculum
effects, non-monotonic LR schedules) that are missed by existing scaling laws.

**Claim 3**: The search process itself is 100x more sample-efficient than random/Bayesian search
due to LLM-guided mutations that encode "research intuition."

### Search Space

| Dimension | Range | Granularity | Why |
|-----------|-------|-------------|-----|
| Data mix (5 domains) | [0, 1] simplex | Continuous | DoReMi showed this matters |
| Filter aggressiveness | [0.01, 1.0] retention rate | Continuous | Bitter Lesson insight |
| Peak learning rate | [1e-4, 1e-3] | Log-uniform | Standard HPO |
| Warmup steps | [500, 5000] | Integer | Affects stability |
| LR decay shape | {cosine, linear, WSD, inv_sqrt} | Categorical | Under-explored |
| Batch size schedule | {constant, linear_ramp, step_ramp} | Categorical | GPT-4 uses ramp |
| Sequence length curriculum | {fixed, linear_ramp, staged} | Categorical | Recent papers suggest this helps |
| Weight decay | [0.01, 0.3] | Log-uniform | Interacts with LR |
| Data curriculum (phase transitions) | When to shift mix ratios | Structured | Novel contribution |

### Experimental Design

#### Phase 1: Proxy Search (190M model, ~100 GPU-hours)

**Goal**: Run MCGS on 190M model, produce ranked recipe list.

**Setup**:
- Model: TransformerConfig.olmo2_190M (768 dim, 12 layers)
- Training: 5000 steps per trial (~5B tokens per trial at 1M tokens/step)
- MCGS cycles: 50-100 cycles
- Eval: C4 val loss + Fineweb-Edu val loss + 3 downstream tasks (ARC-Easy, PIQA, HellaSwag)
- Budget per trial: ~1 GPU-hour on H200
- Total compute: ~100 GPU-hours

**Deliverable**: Top-10 discovered recipes with full trajectory data.

#### Phase 2: Transfer Verification (3B model, ~500 GPU-hours)

**Goal**: Verify that top recipes from 190M transfer to 3B.

**Setup**:
- Model: TransformerConfig.olmo2_3B (3328 dim, 16 layers)
- Run top-3 discovered recipes + OLMo-2 official + random baseline
- Training: 10B tokens each (10000 steps at 1M tokens/step)
- Eval: Same metrics as Phase 1

**Key question**: Does recipe ranking at 190M correlate with ranking at 3B? (Spearman rho)

**Deliverable**: Transfer correlation plot (Fig 2 of paper), best 3B recipe identified.

#### Phase 3: Full-Scale Training (3B, ~2000 GPU-hours)

**Goal**: Full 60B token training with best recipe, then downstream eval.

**Setup**:
- Best recipe from Phase 2, full 60000 steps
- Compare with OLMo-2 3B official checkpoint
- Eval: Full benchmark suite (MMLU, GSM8K, HumanEval, ARC, PIQA, etc.)
- Post-train for Kaggle NeMo Reasoning (LoRA SFT) — same pipeline for both models

**Deliverable**: Main results table, Kaggle submission.

#### Phase 4: 7B-30B Transfer (compute-dependent, ~10K+ GPU-hours)

**Goal**: The "100x scale" headline.

**Setup**:
- Apply discovered recipe to 7B and 13B (if compute allows, 30B)
- Compare with OLMo-2 official at same scale
- Measure whether the gap grows, shrinks, or stays constant with scale

**Deliverable**: The scaling transfer plot that makes the paper.

### Ablation Studies

1. **Search method ablation**: MCGS+LLM vs MCGS+random vs Bayesian vs Random
2. **Proxy fidelity**: How many 190M trials needed before 3B recipes become reliable?
3. **Which dimensions matter most**: SHAP-style feature importance across search dimensions
4. **Bitter Lesson connection**: Plot discovered filter-aggressiveness vs model size → does MCGS
   independently discover that larger models need less filtering?

### Expected Paper Structure

```
1. Introduction
   - Pretraining recipes are hand-designed; we automate the search
   - Key result: X% improvement transfers across 100x scale

2. Related Work
   - Scaling laws (Chinchilla, Scaling Data-Constrained)
   - Data curation (DoReMi, D4, DCLM, Bitter Lesson paper)
   - Transfer of hyperparameters (muTransfer)
   - AutoML for deep learning (PBT, ASHA)

3. Method: MCGS for Pretraining Recipe Search
   - Search space definition
   - LLM-guided mutations
   - Proxy-to-target transfer protocol

4. Experiments
   - Phase 1: Search on 190M (search trajectory, convergence)
   - Phase 2: Transfer to 3B (correlation analysis)
   - Phase 3: Full training + downstream eval
   - Phase 4: 7B/30B transfer (if compute allows)

5. Analysis
   - What recipes does MCGS discover? (qualitative)
   - Bitter Lesson connection (filter aggressiveness vs scale)
   - Search efficiency (LLM vs baselines)
   - Failure modes and when transfer breaks

6. Discussion & Conclusion
   - Release: full codebase + all recipes + checkpoints
   - "100x AutoResearch" vision
```

---

## Option B: "Pretraining Data Mixtures as a Search Problem: From Proxy Models to Kaggle Gold"

### Positioning

- **Venue**: EMNLP / ACL / NeurIPS (datasets & benchmarks track)
- **Audience**: Data curation researchers, Kaggle community, practitioners
- **Comparison papers**: DoReMi, SlimPajama, DataComp, DCLM
- **Novelty**: First to connect proxy-scale data mix optimization to downstream Kaggle competition performance

### Core Claims

**Claim 1**: Automated data mix search on proxy models (190M) finds domain ratios that consistently
outperform expert-designed mixtures (OLMo-2, Llama-3) on downstream reasoning tasks.

**Claim 2**: The optimal data mix is NOT static — it depends on training phase, and a learned
curriculum (e.g., more code/math later in training) outperforms any fixed ratio.

**Claim 3**: Proxy-discovered mixes transfer to 3B scale and produce state-of-the-art results on
the Kaggle NeMo Reasoning competition after standard post-training.

### Search Space (Focused on Data)

| Dimension | Range | Why |
|-----------|-------|-----|
| Web ratio | [0.2, 0.8] | Dominant source, floor effect below 0.2 |
| Code ratio | [0.05, 0.4] | Critical for reasoning (Bitter Lesson: code helps math) |
| Math ratio | [0.02, 0.3] | Direct signal for reasoning |
| Books/academic | [0.0, 0.2] | Quality vs diversity tradeoff |
| Filter threshold (per domain) | [0.01, 1.0] | Bitter Lesson: how much to filter |
| Phase transitions | After N% of training, shift ratios | Curriculum learning |
| Quality scorer | {perplexity, classifier, none} | Meta-parameter of filtering |

**Fixed (not searched)**: LR=3e-4, warmup=2000, cosine decay, batch=256. This isolates the data
effect from optimization noise.

### Experimental Design

#### Phase 1: Data Mix Search (190M, ~80 GPU-hours)

**Goal**: Find optimal static mix + optimal curriculum (dynamic mix).

**Setup**:
- Model: 190M, 5000 steps per trial
- MCGS cycles: 40 (static mix) + 40 (curriculum)
- Eval: Val loss decomposed by domain + downstream reasoning (ARC, PIQA)
- Constraint: All mixes must sum to 1.0

**Deliverable**: Pareto frontier of {val_loss, downstream_reasoning} across mixes.

#### Phase 2: Bitter Lesson Integration (190M + 3B)

**Goal**: Test whether less filtering + more data beats aggressive filtering + less data.

**Setup**:
- Take top-3 mixes from Phase 1
- For each, run 3 filter levels: {aggressive (2%), moderate (20%), minimal (80%)}
- Train at both 190M and 3B scale (shorter runs: 2000 steps)
- Compare: does the Bitter Lesson hold for REASONING tasks too?

**Deliverable**: The "filter aggressiveness vs model size" plot, specifically for reasoning downstream.

**Novel angle**: The Stanford paper evaluates on generic benchmarks. We test on REASONING specifically.
It's plausible that for reasoning, filtered math/code data remains important even at scale.

#### Phase 3: Full Training + Kaggle (3B, ~2000 GPU-hours)

**Goal**: Train 3B with best-discovered mix, post-train for Kaggle.

**Setup**:
- Best recipe: full 60B token run on 3B
- Compare: OLMo-2 recipe, Llama-3 recipe (estimated from paper), our discovered recipe
- Post-train: Same LoRA SFT pipeline on all → Kaggle submission

**Deliverable**: Kaggle leaderboard score + paper results table.

#### Phase 4: Analysis & Open-Source

- Release all mixes as YAML configs
- Release training curves showing where mix changes matter most
- Release the search trajectory (which mutations succeeded/failed)

### Key Experiments for the Paper

#### Experiment 1: Static Mix Optimization

| Mix Name | Web | Code | Math | Books | Academic | 190M Loss | 3B Loss | Kaggle |
|----------|-----|------|------|-------|----------|-----------|---------|--------|
| OLMo-2 | 55% | 20% | 10% | 8% | 7% | X.XX | X.XX | X.XX |
| Llama-3 (est) | 60% | 25% | 10% | 3% | 2% | X.XX | X.XX | X.XX |
| MCGS Best (static) | ?% | ?% | ?% | ?% | ?% | X.XX | X.XX | X.XX |

#### Experiment 2: Curriculum vs Static

| Strategy | Description | 190M Loss | 3B Loss | Kaggle |
|----------|-------------|-----------|---------|--------|
| Static best | Fixed optimal ratio | X.XX | X.XX | X.XX |
| Curriculum: math-late | Start web-heavy, shift to math | X.XX | X.XX | X.XX |
| Curriculum: code-early | Inject code first for structure | X.XX | X.XX | X.XX |
| MCGS curriculum | Discovered phase transitions | X.XX | X.XX | X.XX |

#### Experiment 3: Bitter Lesson for Reasoning

| Filter Level | Data Amount | 190M Loss | 190M Reason | 3B Loss | 3B Reason |
|-------------|-------------|-----------|-------------|---------|-----------|
| Aggressive (2%) | 5B tokens | X.XX | X.XX | X.XX | X.XX |
| Moderate (20%) | 48B tokens | X.XX | X.XX | X.XX | X.XX |
| Minimal (80%) | 192B tokens | X.XX | X.XX | X.XX | X.XX |
| None (100%) | 240B tokens | X.XX | X.XX | X.XX | X.XX |

**Hypothesis**: For reasoning specifically, the Bitter Lesson crossover happens LATER (more
filtering helps longer) because reasoning requires higher quality signal. This would be a
novel finding that refines/nuances their paper.

#### Experiment 4: Search Efficiency

| Method | Trials to 5% improvement | Trials to 10% improvement |
|--------|--------------------------|---------------------------|
| Random search | X | X |
| Bayesian (TPE) | X | X |
| MCGS (random mutations) | X | X |
| MCGS (LLM mutations) | X | X |

### Expected Paper Structure

```
1. Introduction
   - Data mixing is under-explored as a search problem
   - We automate it and show downstream gains (Kaggle)

2. Related Work
   - Data curation: DoReMi, D4, DCLM, DataComp
   - Bitter Lesson for Data Filtering (Stanford 2025)
   - Data curricula: recent work on phased training
   - AutoML: hyperparameter search applied to data

3. Method
   - MCGS for data mix optimization
   - Curriculum as a search dimension
   - Proxy-to-target protocol

4. Experiments
   - Static mix optimization (Table 1)
   - Curriculum discovery (Table 2)
   - Bitter Lesson for reasoning (Table 3, key contribution)
   - Scale transfer: 190M → 3B
   - Downstream: Kaggle NeMo Reasoning

5. Analysis
   - What mixes does MCGS discover? (shifts toward code/math)
   - When does curriculum help? (phase transition visualization)
   - Bitter Lesson refined: reasoning needs more curation than general LM

6. Discussion
   - Open-source recipes
   - Practical guidelines for practitioners
```

---

## Comparison: Which to Do First?

| Criterion | Option A | Option B |
|-----------|----------|----------|
| Compute needed (MVP) | ~2500 GPU-hours | ~2000 GPU-hours |
| Compute for full paper | ~10K+ GPU-hours | ~3000 GPU-hours |
| Time to first results | 4-6 weeks | 3-4 weeks |
| Risk | Medium-high (transfer may not work) | Low-medium (mix optimization usually works) |
| Impact ceiling | Very high (NeurIPS oral) | High (NeurIPS poster / EMNLP best paper) |
| Kaggle integration | Secondary deliverable | Primary deliverable |
| Novelty | "100x automated research" narrative | "Bitter Lesson for reasoning" counter-example |
| Dependencies | Needs Option B results as subset | Self-contained |
| Open-source appeal | Complete system release | Reproducible recipes release |

### Recommended Strategy

**Do Option B first (Weeks 1-8)**, then extend to Option A (Weeks 9-16):

1. Option B's data mix search is a **strict subset** of Option A's full recipe search
2. Option B gives you Kaggle results fast (老板的需求)
3. Option B's "Bitter Lesson for Reasoning" angle is timely (their paper just came out)
4. If Option B's mixes transfer 190M → 3B, that's already evidence for Option A's scale claim
5. Option A then adds: LR/schedule search + 7B/30B transfer on top

---

## Infrastructure Needed

### Already Built (in this repo)

- [x] OLMoCoreBackend implementing TrainingJobRunner protocol
- [x] MCGS search algorithm with LLM-guided mutations
- [x] Config translator: workspace YAML → OLMo-core training scripts
- [x] Script generator: produces correct olmo-core API calls
- [x] K8s PyTorchJob manifests for H200 cluster
- [x] Data preparation pipeline (prepare_data_3b.py)
- [x] Seed workspaces: olmo_3b_pretrain, olmo_core_pretrain (190M)
- [x] NeMo MAS post-training pipeline (for Kaggle downstream eval)

### Needs Building

- [ ] MCGS integration with OLMoCoreBackend (wire run_trial into search loop)
- [ ] Eval harness: automated downstream benchmark suite after each trial
- [ ] Data mix as continuous search dimension (currently discrete workspace files)
- [ ] Filter-aggressiveness parameter (integrate with data prep pipeline)
- [ ] Curriculum support: phase-based data mix transitions during training
- [ ] Scaling law fitting: predict transfer from proxy results
- [ ] Experiment tracking: W&B or similar for 100+ trials
- [ ] 190M fast-iteration pipeline (< 1 hour per trial on single node)

### Compute Estimate

| Phase | GPU-hours (H200) | Cost @ $3/GPU-hr | Timeline |
|-------|------------------|-------------------|----------|
| Phase 1 (190M search) | 100 | $300 | Week 1-3 |
| Phase 2 (3B transfer) | 500 | $1,500 | Week 4-6 |
| Phase 3 (3B full train) | 2000 | $6,000 | Week 7-9 |
| Phase 4 (7B, optional) | 5000 | $15,000 | Week 10-12 |
| Phase 5 (30B, stretch) | 30000 | $90,000 | Needs sponsor |
| **Total (Minimum viable paper)** | **2600** | **$7,800** | **9 weeks** |

---

## Connection to Guanghaw's Work

The pretraining recipe search directly accelerates post-training research:

1. **Better base model → easier post-training**: If MCGS finds a recipe that produces a 3B model
   with stronger reasoning foundations, Guanghaw's LoRA SFT will start from a better point.

2. **Shared evaluation infrastructure**: Both need Kaggle NeMo Reasoning as downstream metric.
   Building this once serves both threads.

3. **Data mix insights transfer**: If MCGS discovers that "more math/code during pretraining"
   helps reasoning, that insight informs what post-training data to emphasize too.

4. **Joint paper potential**: "From pretraining recipe to Kaggle gold: end-to-end automated
   optimization" could be a joint paper if both threads produce complementary results.

---

## Next Steps (Immediate Actions)

1. **Wire MCGS → OLMoCoreBackend**: Make the search loop actually runnable
2. **Build 190M fast-trial pipeline**: Target < 45 min per trial on 8 GPUs
3. **Define eval suite**: C4 loss + 3 downstream tasks, automated after each trial
4. **First experiment**: 10 manual recipe variants on 190M, validate the eval pipeline
5. **First MCGS run**: 20 cycles on 190M, data mix only (Option B Phase 1 start)
