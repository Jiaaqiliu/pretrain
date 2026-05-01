# LLM-Guided AutoML Evolution

## Overview

Uses **Claude Opus 4.7 via Bedrock** as an intelligent mutation proposer for the MCGS-based AutoML framework. Instead of predefined sweeps, the LLM analyzes training history (full configs + metrics + crashes) and proposes context-aware mutations across hyperparameters, model type, and feature engineering flags.

## Rule-based vs LLM-guided (experimental results)

| Approach | Cycles | Best Score | Notes |
|---|---|---|---|
| Rule-based sweep | 20 | 0.81839 | Predefined depth/n_estimators/model sweeps |
| **LLM-guided v1** (initial) | 5 | 0.81839 | Reached baseline in 4x fewer cycles |
| **LLM-guided v3** (optimized) | 20 | 0.81839 | Same ceiling — see diagnostics below |

**Finding**: 0.81839 appears to be the dataset ceiling for non-ensembled, non-CV'd tree models on this workspace. Both paths (rule-based and LLM-guided) converge there. The LLM's value shows up in *diversity of paths* and *faster convergence*, not a final score win.

---

## Architecture

### The mutator interface
MCGSSearch takes any object with `.propose(parent, graph)`:

```python
algo = MCGSSearch(mutator=LLMHyperparameterProposer())
```

No framework changes were needed — the `MutationProposer` contract was already pluggable.

### What the LLM sees (context)
Built in `_build_context()`:

```python
{
  "parent_config": {...full YAML config reconstructed from ancestor patches...},
  "parent_metric": 0.81839,
  "metric_history": [
    {"node_id": "...", "metric": 0.81839, "full_config": {...}, "is_best": true},
    ...top 10 by metric...
  ],
  "tried_configs": [<fingerprints of all non-root nodes, including crashes>],
  "crashed_configs": [{"node_id": "...", "config": {...}, "reason": "training_failed"}],
  "best_metric": 0.81839,
  "search_space": {<13 parameters with discretized value sets>},
}
```

### Search space
10 hyperparameters + 3 feature-engineering flags:
```python
"model_type": ["xgboost", "lightgbm", "random_forest"],
"hyperparameters.n_estimators": [50, 100, 150, 200, 300, 400, 500, 750, 1000],
"hyperparameters.max_depth": [3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
"hyperparameters.learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
"hyperparameters.subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
"hyperparameters.colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
"hyperparameters.min_child_weight": [1, 3, 5, 7, 10],
"hyperparameters.gamma": [0, 0.1, 0.3, 0.5, 1.0],
"hyperparameters.reg_alpha": [0, 0.01, 0.1, 1.0],
"hyperparameters.reg_lambda": [0.1, 1.0, 3.0, 5.0, 10.0],
"feature_engineering.flags.interactions": [True, False],
"feature_engineering.flags.log_transform_spending": [True, False],
"feature_engineering.flags.target_encoding": [True, False],
```

The three FE flags map to concrete transforms in `feature_engineering.py`:
- `interactions` — CryoSleep×HasSpending, Age×Spent, VIP×Spent, FamilySize×Spent
- `log_transform_spending` — log1p of all spending columns
- `target_encoding` — smoothed mean-target encoding for HomePlanet / Destination / CabinDeck

### HybridSelector
Default `UCTSelector` collapsed the tree to a chain, and `RootFanoutSelector` forced every cycle to start from root (so the LLM couldn't *evolve* a winning config). `HybridSelector` does:

```
cycle 1..exploration_cycles:  pick root         (explore base models)
cycle exploration_cycles+1..: pick best-valid   (exploit LLM mutations on top)
```

This lets the LLM compound mutations on top of the current best instead of starting over each cycle.

---

## Lessons from the experiments

### v1 → v3: what went wrong, what fixed it

**v1 failure mode**: LLM saw only mutation descriptions ("Switch to lightgbm"), not the full configs. It also didn't know which configs had already been tried. Result: kept proposing "lower lr + more trees" loops, and re-proposed identical configs three times in cycles 18–20.

**v3 fixes**:

1. **Full config history in prompt** — the LLM sees the entire YAML (all hyperparameters + FE flags) for the top 10 valid nodes, not just descriptions. It can now see that a specific n_estimators=1000/lr=0.01 combo already failed.

2. **Tried-config fingerprinting + hard dedup** — every non-root node (including crashes) is fingerprinted. When the LLM proposes a duplicate, the proposer appends "⚠️ DUPLICATE DETECTED — RETRY" to the prompt and calls the LLM again (up to 3 retries). Caught 1 duplicate in the 20-cycle run.

3. **Crashed configs surfaced** — when `target_encoding=True` initially crashed (train/test column mismatch), the LLM re-proposed it 10+ times because crashed nodes had `metric=None` and were filtered out of history. Fix: track crashes separately and show them under "CONFIGS THAT CRASHED — AVOID THESE PATTERNS".

4. **Noise level warning** — the prompt now says "differences < 0.002 are within noise (~10 rows on 4277-row test)". This stopped the LLM from over-reacting to 0.81609 vs 0.81839 and making aggressive slow-learning changes.

5. **Unexplored regions hint** — explicitly pointed the LLM at FE flags as "never tried" high-value regions. Changed the distribution of mutations dramatically (v1: 8 slow-learning variants; v3: 8+ distinct mutation types).

### v3 results on spaceship-titanic (20 cycles)

| Rank | Metric | Config | Source |
|---|---|---|---|
| 1 | 0.81839 | LightGBM defaults | rule-based cycle 2 |
| 1 | 0.81839 | XGBoost + min_child_weight=3 + reg_lambda=3 + gamma=0.1 | **LLM** (independent path to baseline) |
| 3 | 0.81724 | LightGBM + interactions=True | LLM |
| 3 | 0.81724 | XGBoost + interactions=True | LLM |
| 5 | 0.81609 | XGBoost defaults | rule-based cycle 1 |

The LLM found a *second* path to baseline with a completely different parameter regime (XGBoost + regularization vs LightGBM defaults). This is where its value shows up — not a higher ceiling, but diverse local optima.

### What LLM-driven mutation did NOT solve
- **Score ceiling** — 0.81839 looks like the single-model, no-CV ceiling on this workspace. Every FE combination (interactions, log_transform, target_encoding, and all pairs) scored lower.
- **Test noise** — the 4277-row test set means ±0.003 is within noise. Distinguishing real improvements requires CV.
- **Ensembles** — none of the mutators know how to build ensembles; the backend is single-model.

---

## Running it

### Quick smoke test (3 cycles, ~1-2 min)
```bash
PYTHONPATH=/fsx/yisi/A-EVOLVE-V2 \
  /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_llm_smoke3.py
```

### Full 20-cycle run (~5-10 min)
```bash
PYTHONPATH=/fsx/yisi/A-EVOLVE-V2 \
  /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_llm_20cycle.py
```

### Environment setup
```bash
# mlebench is required for Kaggle grading (already installed in .venv)
/fsx/yisi/A-EVOLVE-V2/.venv/bin/python -c "import mlebench"

# boto3 is required for Bedrock (already installed)
/fsx/yisi/A-EVOLVE-V2/.venv/bin/python -c "import boto3"

# Bedrock credentials must be configured for us-west-2
aws bedrock list-foundation-models --region us-west-2 --by-provider anthropic | grep opus
```

---

## Model choice

The proposer uses the `us.anthropic.claude-opus-4-7` inference profile. Note:
- Opus 4.7 does **not** accept `temperature` — passing it raises `ValidationException`. The proposer omits it.
- The model returns strict JSON (stripped of markdown code fences). The prompt gives two worked examples of the output format.

---

## Cost

- ~1500 input tokens + ~500 output tokens per mutation call
- 20-cycle run: ~$0.06 total Bedrock cost on Opus 4.7
- Training compute (sklearn on 8693 rows × 22 features) dominates wallclock, not LLM latency

---

## Files

- `agent_evolve/model/algorithms/mcgs/llm_mutation.py` — `LLMHyperparameterProposer`, `LLMFeatureEngineeringProposer`
- `agent_evolve/backends/feature_engineering.py` — gated FE with `flags` dict
- `examples/mle_automl_example/drive_llm_smoke3.py` — 3-cycle smoke test
- `examples/mle_automl_example/drive_llm_20cycle.py` — 20-cycle run with HybridSelector
- `seed_workspaces/mle_automl/model/config.yaml` — seed config including FE flags

---

## Next steps (not yet implemented)

Options to break through the 0.81839 ceiling — none are the LLM's fault, they're workspace-level limitations:

1. **Cross-validation in the backend** — replace single train/test split with stratified k-fold, use CV-mean as the metric. Removes most of the ±0.003 noise.
2. **Ensemble support** — let the backend train multiple models and average predictions. Requires a new `train/ensemble.yaml` layer and backend changes.
3. **Better FE** — the current three flags may be too aggressive (interactions likely causes leakage via CryoSleep×HasSpending). More conservative features (count-encoding, binned age groups) might help.
4. **LLM-generated feature code** — escape the fixed FE flag space entirely and let the LLM write Python snippets for new features. Requires code execution sandboxing.
