# Skill: minhash_dedup

When to use: before mixing a new `distill_batch` into the final
training set — confirm it isn't ~95% the same prompts you already
trained on. Also: when an audit flags high near-dup overlap with
prior batches.

## Inputs

- One or more JSONL paths to dedupe (typically a new
  `distill_batch` plus the current `data/final/train.jsonl`).
- The dedup key: usually `prompt_rendered`. For Nemotron, the
  `data/recipes/default.yaml` filter `dedup_by:
  prompt_and_source_hash` uses the prompt+source pair as key.
- Threshold: Jaccard similarity above which two rows are
  considered duplicates. Default 0.85.

## Procedure

1. Decide the key: prompt-only (`prompt_rendered`) for cross-source
   dedup, or `prompt_and_source_hash` for within-source. Match
   what `data/recipes/default.yaml` declares.
2. `minhash_dedup(input_path=<new batch>, key_field=<key>,
   threshold=0.85)` — produces a deduped JSONL and a report of
   collisions. (For multi-input cases, concat first then dedup.)
3. Inspect the report:
   - `kept`: rows that survived
   - `duplicates`: rows dropped, with their nearest-neighbor id in
     the keep set
   - `overlap_with_existing`: if compared against
     `data/final/train.jsonl`, fraction of new batch that
     duplicated existing data
4. If `overlap_with_existing > 0.5`, the new batch is mostly
   redundant. Keep the deduped version but flag this in the
   `dataset_snapshot` body — ResearchScientist may want to know the next
   distill should pull from a different prompt source.

## Output

This is a procedure, not a record-producing skill. The output is
a deduped JSONL path used downstream by `mix_by_curriculum`. The
relevant memory record is the `dataset_snapshot` written by
`mix_by_curriculum`, which references the dedup numbers.

If you find something surprising worth surfacing (e.g. two batches
from supposedly different sources are 95% identical, suggesting an
upstream bug), write a `failed_attempt` or `breakthrough`:

```yaml
kind: failed_attempt
title: "Dedup surprise: <batch_A> and <batch_B> are <%> identical"
body: |
  Compared:        <path_A> vs <path_B>
  Dedup key:       <key>
  Threshold:       <T>
  Overlap:         <%>
  Hypothesis:      <e.g. "both batches sampled from prompts.jsonl
                    with the same seed">
  Recommendation:  <e.g. "regenerate batch_B with seed+1">
tags: ["dedup", "anomaly"]
refs: [<batch_A_id>, <batch_B_id>]
```

## Anti-patterns

- Do NOT dedup across categories — even identical-looking prompts
  in different categories may have different gold answers.
- Do NOT lower the threshold below 0.7 without explicit reason —
  too aggressive collapses legitimate paraphrases.
- Do NOT dedup before format-validating; you don't want to keep a
  malformed row over a valid duplicate. Order: validate → dedup →
  mix.
