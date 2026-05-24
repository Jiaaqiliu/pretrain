# Literature Survey: Pretraining Data Mixture Optimization

## Summary of Key Papers and Their Contributions

### 1. DoReMi (Xie et al., NeurIPS 2023)

**Method**: 3-step DRO-based domain weight optimization
1. Train reference model on default mixture
2. Train proxy (280M) with Group DRO — upweights domains with highest excess loss
3. Use final weights to resample for full model (8B)

**Key numbers**: +6.5 pp downstream accuracy; 2.6x fewer training steps to match baseline.

**Limitation**: Static weights (no curriculum); requires discrete domain definitions upfront.

**Our advantage**: MCGS jointly optimizes mix + curriculum + filter, not just static weights.

---

### 2. Data Mixing Laws (Ye et al., ICLR 2025)

**Method**: Fit parametric scaling law predicting loss from mixture proportions, then optimize analytically.

**Key numbers**: R² > 0.97 prediction accuracy; 48% fewer training steps at 1B/100B.

**Limitation**: Assumes smooth functional form; may miss non-linear interactions.

**Our advantage**: MCGS doesn't assume functional form — it discovers the landscape empirically.

---

### 3. RegMix (Liu, Muennighoff et al., ICLR 2025)

**Method**: Train 512 tiny models (1M params, 1B tokens) with random mixtures; fit regression model; optimize.

**Key numbers**: Outperforms 63 alternative mixtures at 1B/25B scale. Uses 10% of DoReMi's compute.

**Key finding**: "Web corpora rather than data perceived as high-quality like Wikipedia have the strongest positive correlation with downstream performance."

**Limitation**: Linear regression may miss interactions; 512 runs still substantial.

**Our advantage**: MCGS is adaptive (later trials informed by earlier ones); requires fewer total trials.

---

### 4. DCLM: DataComp for Language Models (Li et al., 2024)

**Method**: Standardized benchmark for data curation. Key finding: model-based fastText classifier for quality filtering dominates all heuristic approaches.

**Key numbers**: 7B model at 64% MMLU (matching Mistral-7B-v0.3) with 40% less compute than prior SOTA.

**Our integration**: Use DCLM-style model-based filter as the "quality gate" in our pipeline, with retention threshold as a searchable parameter.

---

### 5. Domain Upsampling at End of Training (Blakeney et al., 2024)

**Method**: Two-phase curriculum — web-heavy for 80-90%, then upsample code/math/curated in final 10-20%.

**Key numbers (7B, 1T tokens)**: +6.90 pp MMLU, +8.26 pp GSM8K, +6.17 pp HumanEval.

**Key finding**: Optimal annealing window = 10-20% of total duration. Too early hurts generality.

**Our integration**: This is exactly our CurriculumSchedule with 2 phases. MCGS should discover the optimal transition point and domain ratios for each phase.

---

### 6. Scaling Data-Constrained Language Models (Muennighoff et al., 2023)

**Method**: 400 training runs up to 9B/900B studying data repetition effects.

**Key findings**:
- Up to 4 epochs: negligible loss increase (free repetition)
- Beyond 4 epochs: diminishing returns, model memorizes
- When data-constrained: relaxing quality filters > repeating filtered data
- Code data helps non-code tasks (structured reasoning transfer)

**Our constraint**: Search must respect the 4-epoch ceiling per domain. If a domain is small, its weight is bounded by `domain_size / (4 * total_training_tokens)`.

---

### 7. A Bitter Lesson for Data Filtering (Mohri, Duchi, Hashimoto — Stanford, 2025)

**Method**: Empirical study showing unfiltered CC eventually beats filtered at sufficient scale.

**Key finding**: The crossover point is predictable via scaling laws. At ~10³⁰ FLOPs, filtering becomes provably suboptimal.

**Our angle**: Test whether this holds for REASONING tasks specifically. Hypothesis: reasoning needs more signal, so crossover is later or absent.

---

## Synthesis: Design Principles for Our Algorithm

Based on this survey, our MCGS-based approach should:

1. **Start from DoReMi-like baseline** — use DRO-optimized weights as the initial MCGS root
   (not arbitrary OLMo-2 weights). This gives us a strong starting point.

2. **Search jointly** (unlike any single prior work):
   - Domain ratios (like DoReMi/RegMix)
   - Filter threshold (like DCLM + Bitter Lesson)
   - Curriculum schedule (like Domain Upsampling)
   - Subject to repetition constraint (Muennighoff et al.)

3. **Use domain loss decomposition as feedback** — per-domain val loss tells the mutator
   which domains need more/less weight. This is richer signal than scalar loss.

4. **Bias curriculum search toward late-stage upsampling** — Blakeney et al. shows this
   is where the biggest gains are. Our mutator should have high prior probability of
   creating curricula with reasoning-heavy final phases.

5. **Respect the 4-epoch constraint** — bound domain weights by available unique tokens.
   For small domains (math, books), weights are upper-bounded.

6. **Compare against proper baselines**: DoReMi (DRO), RegMix (regression), static best,
   and manual baselines (OLMo-2, Llama-3).

---

## Positioning Our Contribution

| Prior Work | What they optimize | How they search | Our delta |
|------------|-------------------|-----------------|-----------|
| DoReMi | Domain weights (static) | DRO optimization | + curriculum + filter + LLM-guided |
| Data Mixing Laws | Domain weights | Scaling law fitting | + no functional form assumption |
| RegMix | Domain weights | Random sampling + regression | + adaptive search (fewer trials) |
| Domain Upsampling | Annealing schedule | Manual grid search | + automated + more flexible |
| DCLM | Filter threshold | Fixed | + joint with mixture + compute-aware |
| Bitter Lesson | Nothing (observation) | N/A | + actionable algorithm + reasoning-specific |

**Our unique contribution**: First to jointly and adaptively search (mix + curriculum + filter)
with proxy-to-target transfer verification, validated on reasoning downstream.

---

## Exact Data Ratios from Technical Reports

### OLMo-2 (Allen AI, Jan 2025) — 7B/13B/32B

**Stage 1 (3.9T tokens)**:
| Source | Tokens | % |
|--------|--------|---|
| DCLM-Baseline (web) | 3.71T | 95.1% |
| StarCoder (code) | 83B | 2.1% |
| peS2o (academic) | 58.6B | 1.5% |
| arXiv | 20.8B | 0.5% |
| OpenWebMath | 12.2B | 0.3% |
| Algebraic Stack | 11.8B | 0.3% |

**Stage 2 "Mid-training" (50B for 7B model)**:
- High-quality web (DCLM top 7% + FineWeb-Edu ≥ 2): 47.2%
- Math mix (synthetic + rewritten): 19.6%
- FLAN instructions: 16.6%
- Academic: 5.15%
- Wikipedia/Wikibooks: 7.11%

**Key techniques**: Microannealing (50B test runs); checkpoint souping (avg 3 annealed ckpts);
2-4x math duplication is beneficial; MIND rewriting (natural language > code format for math).

### Llama-3 (Meta, Jul 2024) — 8B/70B/405B, 15T tokens

**Final mix**: General=50%, Math+Reasoning=25%, Code=17%, Multilingual=8%

**Annealing** (final 40M tokens): upsample code+math, context=128K, Polyak averaging.
- 8B: +24 pp GSM8K, +6.4 pp MATH from annealing alone
- 405B: negligible gain (already strong in-context learner)

**Batch size ramp**: 4M → 8M → 16M tokens.

**Filtering**: 3-level dedup (URL, MinHash, line-level >6 occurrences);
fastText + RoBERTa quality classifiers; knowledge classifier for category balance.

### DeepSeek-V3 (Dec 2024) — 671B MoE (37B active), 14.8T tokens

- "Enhanced ratio of math and programming vs V2" (exact ratios undisclosed)
- FIM at 10% rate
- Batch ramp: 3072 → 15360 sequences over first 469B tokens
- Zero loss spikes, zero rollbacks (remarkable stability)
- Cost: $5.6M total for pretraining
- Novel: Multi-Token Prediction, FP8 training at scale

### DCLM Scale Transfer Correlation

Critical finding for our proxy approach:
| Proxy Scale | Correlation with 7B |
|-------------|-------------------|
| 400M | r = 0.838 |
| 1B | r = 0.956 |
| 3B | r = 0.982 |

**This directly validates our approach**: recipe rankings at 190M-3B proxy scale
are highly predictive of performance at 7B+ target scale.

---

## Cross-Lab Patterns (Consistent Across All Labs)

1. **Multi-stage training** with mix transitions is now standard (all labs use 2-3 stages)
2. **Math/code upsampled in later stages** (OLMo: 20% math mid-train; Llama: 25% reasoning final)
3. **Batch size ramp-up is universal** (all labs start small, grow 2-4x)
4. **fastText classifiers** are the dominant quality filter approach
5. **2-4x repetition of math data** is beneficial (OLMo-2 finding)
6. **Model averaging** during annealing improves results (OLMo-2 checkpoint souping, Llama-3 Polyak)
7. **Sequence length curriculum**: start short (4K), extend late (32K-128K)
8. **Well-filtered web alone can outperform diverse mixes** (DCLM finding)
9. **Annealing with domain upsampling** gives massive gains at small model sizes (Llama-3 8B: +24 pp)
10. **Synthetic math data** is increasingly critical (OLMo-2, Qwen2.5, Qwen3)
