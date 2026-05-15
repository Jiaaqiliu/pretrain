---
source: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
scope: priors for SFT-LoRA hyperparameter proposals (lr, scheduler, batching, weight_decay, dropout). Cite when proposing on these levers.
---

# LoRA Fine-tuning Tips

> Source: [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

## Core Hyperparameters

| Hyperparameter | Recommended | Notes |
| --- | --- | --- |
| `learning_rate` | `2e-4` (SFT) / `5e-6` (DPO, GRPO, RL) | Full FT uses smaller LR |
| `num_train_epochs` | 1–3 | Diminishing returns past 3; overfitting risk |
| `r` (rank) | 16 or 32 (common range 8–128) | Larger rank → easier to overfit |
| `lora_alpha` | `r` or `2 * r` | Keep `α / r ≥ 1` |
| `lora_dropout` | 0 (default), 0.1 if overfitting | Unreliable in short training runs |
| `weight_decay` | 0.01 (common) – 0.1 | Don't go too high |
| `warmup_ratio` | 5–10% of total steps | |
| `lr_scheduler_type` | `linear` or `cosine` | |
| `bias` | `"none"` | Training bias terms adds little value |

## Target Modules

Apply LoRA to **all major linear layers** — both attention and MLP:

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

Attention-only significantly underperforms; MLP-only is close to all layers. Don't drop modules to save memory — the savings aren't worth it.

## Alpha and Rank Relationship

Standard LoRA:

$$\hat{W} = W + \frac{\alpha}{r} AB$$

rsLoRA (rank-stabilized, `use_rslora=True`):

$$\hat{W} = W + \frac{\alpha}{\sqrt{r}} AB$$

In practice, use `α = r` or `α = 2r`. rsLoRA is more stable at high rank.

## Batch Size and Gradient Accumulation

```
Effective Batch Size = batch_size × gradient_accumulation_steps
```

Target effective batch size of 16 works well for most tasks. If VRAM is tight, lower `batch_size` and raise `gradient_accumulation_steps` — the product stays the same.

## Train on Completions Only

Mask out the user/prompt portion and compute loss only on assistant responses. Usually gives ~1% accuracy boost, especially noticeable for multi-turn conversations.

## Overfitting

- Reduce epochs (1–3)
- Increase `weight_decay` (0.01 → 0.1)
- Increase `lora_dropout` (0.1)
- Increase effective batch size
- LoRA alpha scaling: multiply α by 0.5 at inference — equivalent to averaging base and fine-tuned weights
- Eval-based early stopping
- Expand / improve dataset quality

## Underfitting

- Tune LR (try up for short runs, down for long runs — test both)
- More epochs (watch val loss)
- Increase `r` and `α` (rank should be ≥ α; use larger rank for small models / complex tasks, typically 4–64)
- Use a more domain-relevant dataset
- Drop `batch_size` to 1 for more aggressive updates
