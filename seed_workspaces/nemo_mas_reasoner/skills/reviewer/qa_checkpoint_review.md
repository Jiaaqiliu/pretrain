# Skill: qa_checkpoint_review

**When to use**: The orchestrator spawned you with a task that names a
Quality Plan slot (e.g. `cp_02_model_ready`, `cp_04_sft_round1`) and asks
you to decide whether the slot is "ready to sign" or needs more
evidence. This is your QA-officer hat; it's distinct from the
data-auditing skills.

**What you produce**: exactly one `checkpoint_review` record, posted via
the `checkpoint_review_suggest` tool. In auto mode you may additionally
call `checkpoint_sign` if your verdict is `ready_to_sign`.

## The four verdicts

- `evidence_attached` — at least one evidence record exists for the
  slot, but the `requires_evidence` list isn't fully covered yet.
  State goes to `pending_evidence`.
- `ready_to_sign` — all `requires_evidence` kinds are present AND the
  evidence you've read looks healthy. In manual mode the slot becomes
  `pending_human`; in auto mode you should call `checkpoint_sign` to
  close it.
- `insufficient` — evidence is present but you can't tell whether
  it's good. Don't guess; ask the orchestrator for a rerun. State
  stays where it is.
- `reject` — evidence looks wrong (e.g. loss NaN, eval metrics
  regressed, data corruption). This reopens the slot; the pipeline
  must re-produce evidence before the slot can close.

## Procedure

1. **Read the task brief carefully.** The orchestrator names the slot
   id and usually cites specific record ids (`refs`) it thinks you
   should review. Those refs are your starting point.

2. **Find all slot-tagged evidence.** The task should list them, but
   double-check via `mem_search` with `tags=["checkpoint:<slot_id>"]`.
   Don't rely on global kind counts — a `profile_run` untagged for
   this slot doesn't help it.

3. **Read each candidate.** Use `mem_get(id)` on each. Don't just scan
   titles. For each evidence kind, ask yourself the question:

   - `profile_run` → is the loss curve descending? Is init loss near
     expected (ln(vocab_size) for random init)? Did the overfit-batch
     test succeed?
   - `training_run` → did it reach the declared `max_steps`? Is the
     final loss reasonable for the data scale? Any NaN/Inf flags?
   - `eval_report` → primary metric above threshold stated in the
     slot's `benchmark_reference.md` context? Error buckets not
     dominated by a single category?
   - `dataset_snapshot` → row count > baseline? Per-source counts
     balanced? Any `filter_error` column suggesting format breakage?
   - `recipe_proposal` → does the diff target the current blocker's
     concern (e.g. cp_03_lora_config needs rank-related fields)?

4. **Pick the verdict.** Use this decision tree:

   ```
   Does the slot have at least one evidence record per
   requires_evidence kind (tagged checkpoint:<slot_id>)?
     ├─ No  → evidence_attached (if some) or skip (nothing at all)
     └─ Yes → is the evidence healthy (your read of the bodies)?
              ├─ Clearly healthy         → ready_to_sign
              ├─ Can't tell / ambiguous  → insufficient
              └─ Clearly unhealthy       → reject
   ```

5. **Post the verdict.** Call:

   ```
   checkpoint_review_suggest(
       slot_id="cp_XX_YY",
       verdict="ready_to_sign",                  # or one of the other three
       reason="profile_run rec_abcd shows init=2.11 (expected 2.03), "
              "overfit batch loss→0.02 in 40 steps, no NaN",
       refs=["rec_abcd", "rec_efgh"],            # the evidence you read
   )
   ```

6. **Auto mode only**: if verdict is `ready_to_sign`, follow up with
   `checkpoint_sign(slot_id, refs=[...])`. The tool revalidates evidence
   before appending the `checkpoint_event{event:signoff}`. If it
   refuses, your review stays on the record.

## Guardrails

- **Never sign a slot you produced evidence for in this same cycle.**
  Wait for a fresh reviewer spawn. This is how we keep QA
  independent from production.
- **Never invent evidence.** If `requires_evidence` isn't covered,
  the right answer is `insufficient`, not `ready_to_sign`.
- **The reason line is what the human sees.** Be specific. "Metrics
  look good" is not useful. "loss=2.11→0.02 in 40 steps (overfit
  test), all eval buckets have >= 50 rows" is useful.
- **Manual mode stops at `checkpoint_review_suggest`.** Don't call
  `checkpoint_sign` — the handler will refuse anyway, but posting a
  clean review record is the atomic unit of your contribution.

## Anti-patterns

- Walking the full memory — you're only judging one slot. Stay
  focused on its `requires_evidence` kinds.
- Treating "has any evidence" as `ready_to_sign`. The fold does that
  promotion on your behalf; your job is to **read** the evidence.
- Writing a long narrative in the reason field. The cockpit truncates
  at ~240 chars. One useful line.
