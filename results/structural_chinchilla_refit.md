# Structural Chinchilla Formula Refit

## Data Points (10 total)

| Model | N (params) | D/N (tokens/param) | alpha (measured) |
|-------|-----------|--------------------:|------------------:|
| Pythia-70M | 7.0e7 | 4261 | 2.60 |
| Pythia-160M | 1.6e8 | 1848 | 2.63 |
| Pythia-410M | 4.1e8 | 740 | 2.73 |
| Pythia-1B | 1.0e9 | 297 | 2.78 |
| Pythia-2.8B | 2.8e9 | 108 | 5.16 |
| Pythia-6.9B | 6.9e9 | 44 | 5.13 |
| Amber-7B | 7.0e9 | 187 | 5.25 |
| OLMo-2-13B | 1.37e10 | 365 | 6.95 |
| OLMo-2-32B | 3.22e10 | 189 | 5.25 |
| K2-65B | 6.53e10 | 21 | 5.09 |

Sources for new data points:
- OLMo-2-13B: stage1-step596057-tokens5001B, alpha_mean=6.953, D/N = 5001e9/13.72e9 = 365
- OLMo-2-32B: stage1-step721901-tokens6056B, alpha_mean=5.251, D/N = 6056e9/32.23e9 = 188
- K2-65B: ministage2_ckpt_374, alpha_mean=5.093, D/N=21 (1.37T tokens / 65.3B params)

---

## Model 1: Original Exponential (Refit on All 10 Points)

**Formula:** `alpha(D/N) = alpha_inf + A * exp(-D/(tau*N))`

**Original parameters (7-point fit):** alpha_inf=2.54, A=3.5, tau=269

**Refit parameters (10-point fit):**
- alpha_inf = 2.3933
- A = 3.1197
- tau = 845.31

**Results:**
- R^2 = 0.480
- RMSE = 1.053

**Residuals:**
| Model | Predicted | Actual | Residual |
|-------|-----------|--------|----------|
| Pythia-70M | 2.413 | 2.60 | +0.187 |
| Pythia-160M | 2.744 | 2.63 | -0.114 |
| Pythia-410M | 3.693 | 2.73 | -0.963 |
| Pythia-1B | 4.589 | 2.78 | -1.809 |
| Pythia-2.8B | 5.139 | 5.16 | +0.021 |
| Pythia-6.9B | 5.355 | 5.13 | -0.225 |
| Amber-7B | 4.894 | 5.25 | +0.356 |
| OLMo-2-13B | 4.419 | 6.95 | +2.531 |
| OLMo-2-32B | 4.888 | 5.25 | +0.362 |
| K2-65B | 5.436 | 5.09 | -0.346 |

**Verdict:** The exponential form with D/N as sole predictor fails badly. The data is NOT well-described by a single curve in D/N space. The fundamental problem: OLMo-2-13B (D/N=365) and Pythia-1B (D/N=297) have similar D/N but wildly different alpha (6.95 vs 2.78). Model size matters independently.

---

## Model 2: Size-Dependent tau

**Formula:** `alpha = alpha_inf + A * exp(-D/(tau(N)*N))` where `tau(N) = tau_0 * (N/N_0)^beta`, N_0=1e9

**Fit parameters:**
- alpha_inf = 2.6848
- A = 2.8623
- tau_0 = 34.16
- beta = 3.0000 (hit upper bound)

**Results:**
- R^2 = 0.880
- RMSE = 0.505

**Assessment:** Better than Model 1, but beta hitting the bound of 3.0 means the model is essentially creating a step function (tau explodes for large N, making the exponential term constant). This is a sigmoid in disguise.

---

## Model 3: Power Law in D/N

**Formula:** `alpha(D/N) = alpha_inf + A / (D/N)^gamma`

**Fit parameters:**
- alpha_inf = 0.0000
- A = 8.5344
- gamma = 0.1226

**Results:**
- R^2 = 0.346
- RMSE = 1.180

**Verdict:** Worse than Model 1. The power law form with D/N as sole predictor cannot capture the data structure.

---

## Model 4: Power Law with Size Dependence

**Formula:** `alpha = alpha_inf + A * (N/N_0)^delta / (D/N)^gamma`

**Fit parameters:**
- alpha_inf = 0.0000
- A = 3.9794
- gamma = 0.0100 (essentially zero)
- delta = 0.1175

**Results:**
- R^2 = 0.640
- RMSE = 0.876

**Assessment:** The gamma~0 means D/N contributes almost nothing -- the model reduces to a pure power of N. Still underfits OLMo-2-13B by 1.85.

---

## Model 5 (BEST): Sigmoid Transition + Linear D/N Correction

**Formula:**
```
alpha(N, D/N) = alpha_s + [delta + c * (D/N)] * sigma(log10(N))

where sigma(x) = 1 / (1 + exp(-(log10(N) - t) / w))
```

**Fit parameters (differential evolution, global optimum):**
- alpha_s = 2.6525 (small-model baseline)
- delta = 2.0741 (size-transition jump)
- t = 9.2291 (transition at N ~ 1.7e9 params)
- w = 0.0699 (very sharp transition)
- c = 0.005013 (D/N slope for large models)

**Results:**
- **R^2 = 0.971**
- **Adjusted R^2 = 0.935**
- **RMSE = 0.248**
- Max |residual| = 0.424

**Residuals:**
| Model | Predicted | Actual | Residual |
|-------|-----------|--------|----------|
| Pythia-70M | 2.652 | 2.60 | -0.052 |
| Pythia-160M | 2.653 | 2.63 | -0.023 |
| Pythia-410M | 2.653 | 2.73 | +0.077 |
| Pythia-1B | 2.782 | 2.78 | -0.002 |
| Pythia-2.8B | 5.157 | 5.16 | +0.003 |
| Pythia-6.9B | 4.947 | 5.13 | +0.183 |
| Amber-7B | 5.664 | 5.25 | -0.414 |
| OLMo-2-13B | 6.556 | 6.95 | +0.394 |
| OLMo-2-32B | 5.674 | 5.25 | -0.424 |
| K2-65B | 4.832 | 5.09 | +0.258 |

---

## Model Comparison Summary

| Model | Parameters | R^2 | Adj R^2 | RMSE | AIC |
|-------|-----------|------|---------|------|-----|
| Original formula (fixed) | 3 | 0.261 | -0.109 | 1.255 | 38.9 |
| Refit Exponential | 3 | 0.480 | 0.220 | 1.053 | 35.4 |
| Power Law (D/N only) | 3 | 0.346 | 0.019 | 1.180 | 37.7 |
| Sigmoid + Exp correction | 6 | 0.951 | 0.854 | 0.322 | 17.7 |
| **Sigmoid + linear D/N** | **5** | **0.971** | **0.935** | **0.248** | **10.5** |
| Sigmoid + log(D/N) | 5 | 0.934 | 0.851 | 0.375 | 18.8 |

**Winner: Sigmoid + linear D/N correction** (lowest AIC, highest adjusted R^2)

---

## Key Findings

### 1. The Original Formula Is Fundamentally Wrong in Structure

The original Structural Chinchilla formula `alpha(D/N) = 2.54 + 3.5*exp(-D/(269*N))` assumes alpha is a function of D/N alone. This is **rejected** by the new data (R^2=0.26 on 10 points).

The critical counter-example: OLMo-2-13B at D/N=365 has alpha=6.95, while Pythia-1B at D/N=297 has alpha=2.78. Same D/N ratio, completely different alpha. **Model size (N) is an independent variable.**

### 2. The Corrected Formula (Recommended)

```
alpha(N, D/N) = 2.65 + [2.07 + 0.005 * (D/N)] * sigma((log10(N) - 9.23) / 0.070)
```

Simplified interpretation:
- **Small models (N < 1.7B):** alpha ~ 2.65 regardless of training
- **Large models (N > 1.7B):** alpha ~ 4.73 + 0.005 * (D/N)

The transition is extremely sharp (occurs between ~1B and ~2.8B parameters).

### 3. Physical Interpretation

For large models, **more training increases alpha** (opposite to the original formula's prediction). This means extended training causes weight matrices to become more rank-deficient -- fewer singular values dominate. This is consistent with models developing increasingly structured, low-rank representations as they extract more information from the data.

### 4. Outlier Analysis

**OLMo-2-13B is a soft outlier** (residual +0.39 in best model, but +1.77 if using pure sigmoid without D/N term). This is the model that forces the D/N-dependence into the formula.

Leave-one-out analysis:
- Excluding OLMo-2-13B: pure sigmoid achieves R^2=0.998 on remaining 9 points
- Excluding OLMo-2-13B: best model leaves residual=+1.54 (strong deviation)
- Excluding Amber-7B: R^2 improves from 0.971 to 0.980

Possible explanations for OLMo-2-13B's high alpha:
1. **Genuine over-training effect:** D/N=365 is exceptionally high for a 13B model (OLMo-2-32B only reaches D/N=189). Extended training drives alpha up monotonically.
2. **Architecture/recipe differences:** OLMo-2 uses different training recipes (stage1/stage2 annealing) that may affect spectral structure differently than Pythia's straightforward training.
3. **Measurement from the training trajectory supports (1):** alpha grows steadily from 4.25 at D/N=4 to 6.95 at D/N=365, suggesting this is a real training-duration effect.

### 5. Limitations

- Only 10 data points total (5 free parameters in best model leaves only 4 degrees of freedom)
- The D/N coefficient (0.005) is determined largely by a single model (OLMo-2-13B)
- No models in the 1-2.8B transition region besides Pythia-1B (which lies on the small-model side)
- Different model families (Pythia, Amber, OLMo-2, K2) may have systematic architecture effects conflated with size

### 6. Recommendations for Future Work

1. **Critical data point needed:** A model in the 1.5-2.5B range trained to high D/N (>200) would disambiguate the transition
2. **OLMo-2-32B trained longer** would test whether the D/N coefficient holds for 32B models
3. **Multiple checkpoints** at the same model size but different D/N would directly measure the D/N slope per model
4. Consider fitting separate families (Pythia vs OLMo-2 vs K2) to check for systematic offsets
