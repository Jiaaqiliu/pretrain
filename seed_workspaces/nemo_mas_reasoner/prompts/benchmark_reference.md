# Benchmark reference: NVIDIA Nemotron Model Reasoning Challenge (Kaggle)

This file is the human-curated ground truth about how the benchmark is
scored and what kinds of failures matter. Every role gets it appended
to their system prompt. Edit this file when you discover something new
about the eval — do not let an LLM rewrite it without review.

Source: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

## Primary metric

Simple accuracy: proportion of test rows whose predicted answer
matches gold. The judge:

1. Extracts a final answer from the model's response, prioritizing
   content inside `\boxed{...}`. If no box is present, the judge falls
   back to other heuristic patterns (e.g. last numeric value, regex
   patterns).
2. A row is correct if the extracted answer matches gold either as an
   exact string OR within relative tolerance 1e-2 for numeric answers.

Reference implementation: `kaggle.com/code/metric/nvidia-nemotron-metric`.

Score range: [0, 1]. Top of leaderboard at the time of writing: ~0.87
(3-way tie); ~7 teams at 0.86.

Implications:

- Boxing is the cleanest way to surface the answer; relying on the
  fallback is brittle.
- For numeric answers, small floating-point drift is forgiven.
- For categorical / string answers, exact match required (so units,
  thousands separators, LaTeX wrappers can break match).

## Submission contract (HARD FACTS — hard-coded by the host)

**Submission is NOT a CSV.** Submission is `submission.zip` containing
a LoRA adapter for the Nemotron-3-Nano-30B base model.

- Base model (frozen): `NVIDIA Nemotron-3-Nano-30B (A3B BF16)`.
- Max LoRA rank: 32.
- Adapter must include `adapter_config.json`.
- The host loads base + adapter into vLLM and scores the dev/test
  CSV server-side.
- Max 5 submissions per day; pick 2 final selections.

## Inference contract (host-side, vLLM, do not deviate during distill / eval)

- temperature: **0.0** (deterministic decoding)
- top_p: 1.0
- max_tokens: **7680** (generation cap per response)
- max_model_len: **8192** (prompt + generation budget)
- max_lora_rank: 32
- max_num_seqs: 64
- gpu_memory_utilization: 0.85

A response that exceeds 7680 generated tokens is truncated and almost
always misses the box → silent failure.

**Consequence for self-distill (DataWorker / SolverDistiller):**
because eval is deterministic, the same checkpoint + same prompt
produces the same output every time. Self-distill via rejection
sampling at temp=0.0 yields one trace per prompt — to amplify
coverage you must (a) use different prompts, or (b) explicitly raise
distill temperature above 0.0 (and accept the distribution shift
between distill data and inference). Default to (a).

## Compute environment

The host evaluates on Google Cloud G4 VMs with NVIDIA RTX PRO 6000
Blackwell GPUs. Our training runs may use different hardware (H200,
RTX PRO 6000, etc.), but the **inference contract above is host-side
and immutable**. The platform's SFT/RL runners under
`agent_evolve/model/runners/stages/` target the training GPU; never
assume training-time GPU is the same as eval-time GPU.

## Reasoning categories

The official competition page does NOT enumerate categories — only
states the dataset covers "logical reasoning puzzles requiring
identification and application of underlying transformation rules"
across "various domains, such as bit manipulation and algebraic
equations."

Empirically observed categories (from inspecting train.csv — NOT
official; the private test may include unseen rule families):

- bit_manipulation
- cryptarithm
- gravity
- unit_conversion
- roman_numerals
- text_encryption
- equation_transformation

**Implication for Planner + DataWorker**: do not over-fit to these
seven. A recipe that wins on these may regress on a new category in
the private test set. When proposing per-category upsampling, also
keep at least one "generic" data source untargeted to preserve
distribution diversity.

## Error taxonomy (`eval/error_taxonomy.yaml`)

| Bucket | Meaning |
|---|---|
| `format_error` | No `\boxed{}`, judge fallback failed too |
| `wrong_rule` | Model inferred the wrong transformation |
| `partial_rule` | Model found part of the rule but failed composition |
| `answer_extraction_fail` | A box exists but the parser couldn't normalize it (likely judge-side issue, surface and verify) |
| `overlong_reasoning` | Output hit the 7680-token cap with no box |
| `eval_runtime_error` | Infrastructure failure during eval |

Reviewer's `eval_report` records SHOULD break down errors by bucket
and by category — this is what Planner reads to decide what to
change.

## Known sensitivities (update as we learn)

- **CoT length is double-edged**: longer reasoning helps multi-step
  problems but pushes us into `overlong_reasoning`. With max_tokens
  = 7680, training data with completions > ~7000 tokens will
  systematically truncate at eval. Watch the length distribution
  per category.
- **Boxing discipline is fragile**: SFT data with sloppy box markers
  contaminates the model. The recipe filter
  `require_verify_pass: true` (in `data/recipes/default.yaml`)
  exists for this.
- **Per-category upsampling has high leverage**: the cycle-2/cycle-3
  observation that cryptarithm at upsample 12 yielded +0.01 on the
  leaderboard. Don't ignore single-category levers.
- **Deterministic eval blurs CV semantics**: rerunning the SAME
  checkpoint on the SAME split gives the SAME score. CV across seeds
  must vary the **training** seed (data shuffle order, LoRA init),
  not the eval seed. See `skills/trainer/cross_validate_recipe`.

## Timeline + prizes (for context)

- Start: 2026-03-16
- Mid-cutoff (Open Progress Prize): 2026-04-09 (passed)
- Methodology submission: 2026-04-16 (passed)
- Final deadline: see Kaggle page
- Prizes: $25k+5 DGX Sparks (1st), $15k+2 (2nd), $5k+1 (3rd),
  $5k+1 Open Progress, plus 3 Open Contribution Awards
  (Data / RL / Fine-tuning)
- Public Kaggle notebook + writeup REQUIRED for prize eligibility
- Open Contribution Awards require top-10% finish

## External data + license

- External data allowed if "publicly accessible / reasonable cost".
- Winner license: CC BY 4.0 — must open-source training + inference
  code.
