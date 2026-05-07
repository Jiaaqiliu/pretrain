# Nemo_MAS Agent Teams — Current Structure

Snapshot of the multi-agent system that currently drives the Nemotron Reasoning
training pipeline (`seed_workspaces/nemo_mas_reasoner`). Each role is a separate
Claude Code subagent with its own tool allowlist and system prompt; they
communicate via shared typed-record memory and direct `SendMessage` calls.

## Topology

```
                         ┌──────────────────────────────┐
                         │   Human (trace viewer chat)   │
                         │  human_directive records /    │
                         │  sign cp_XX / redirects       │
                         └──────────────┬────────────────┘
                                        │ directive_respond()
                                        ▼
                         ┌──────────────────────────────┐
                         │        ORCHESTRATOR           │
                         │  - reads memory only          │
                         │  - spawns workers, routes QA  │
                         │  - never writes mem records   │
                         │  tools: mem_*, list_slots,    │
                         │  checkpoint_state, start/     │
                         │  current_iteration, TodoWrite │
                         └───┬─────────┬─────────┬──────┘
             spawn_and_run   │         │         │
                             ▼         ▼         ▼
              ┌──────────────┐  ┌────────────┐  ┌──────────────┐
              │   PLANNER    │  │ DATA_WORKER│  │   TRAINER    │
              │ writes:      │  │ writes:    │  │ writes:      │
              │ hypothesis,  │  │ distill_   │  │ training_run,│
              │ recipe_      │  │ batch,     │  │ cv_result,   │
              │ proposal     │  │ dataset_   │  │ submission_  │
              │              │  │ snapshot   │  │ artifact     │
              │ tools: diff_ │  │ tools:     │  │ tools:       │
              │ yaml, render_│  │ teacher_   │  │ launch_      │
              │ recipe_diff, │  │ distill,   │  │ training,    │
              │ read_train_  │  │ batch_     │  │ compute_     │
              │ log, WebFetch│  │ generate,  │  │ stability,   │
              │              │  │ minhash_   │  │ pack_        │
              │              │  │ dedup, mix │  │ submission   │
              └──────┬───────┘  └─────┬──────┘  └──────┬───────┘
                     │                │                │
                     │ refs           │ refs           │ refs
                     ▼                ▼                ▼
                ┌────────────────────────────────────────┐
                │   TYPED-RECORD MEMORY (BM25 + DAG)      │
                │   breakthrough / hypothesis / recipe_   │
                │   proposal / dataset_snapshot /         │
                │   training_run / cv_result / eval_      │
                │   report / data_gap / checkpoint_review │
                └────────────────────────────────────────┘
                                 ▲
                                 │ reads everything; audits + verdicts
                                 │
                         ┌───────┴────────────────────┐
                         │         REVIEWER           │
                         │ 2 hats:                    │
                         │  (a) data/eval analyst     │
                         │  (b) Quality Plan officer  │
                         │ writes: data_audit_finding,│
                         │ profile_run, eval_report,  │
                         │ error_pattern, data_gap,   │
                         │ checkpoint_review,         │
                         │ benchmark_rule, kaggle_    │
                         │ submission_result          │
                         │ tools: run_eval, run_short_│
                         │ training, format_validate, │
                         │ checkpoint_review_suggest, │
                         │ checkpoint_sign,           │
                         │ kaggle_submit              │
                         └────────────────────────────┘
```

## Roles at a glance

| Role           | Decides?                      | Executes?                     | Writes which kinds                                          | Never writes                             |
|----------------|-------------------------------|-------------------------------|-------------------------------------------------------------|------------------------------------------|
| Orchestrator   | Who runs next                 | No                            | — (read-only on memory)                                     | any record                               |
| Planner        | What to change                | No                            | `hypothesis`, `recipe_proposal`                             | training / data / eval                   |
| Data Worker    | No (executes distill specs)   | Teacher / self-distill, mix   | `distill_batch`, `dataset_snapshot`                         | recipes, audits                          |
| Trainer        | No (executes recipes)         | SFT / RL launches, CV reruns  | `training_run`, `cv_result`, `submission_artifact`          | recipes, eval scores                     |
| Reviewer       | Go/no-go on evidence          | Eval, short probes, Kaggle    | `data_audit_finding`, `profile_run`, `eval_report`, `error_pattern`, `data_gap`, `checkpoint_review`, `benchmark_rule` | recipes, training runs |

All workers can additionally write cross-cutting `breakthrough`, `failed_attempt`,
and `checkpoint_event` — but each declares `role=<its_role>` on every
`mem_write`, and the MCP role guard rejects off-whitelist kinds.

## Control flow (one cycle)

1. **Orchestrator** calls `current_iteration` / `start_iteration`, then
   `list_slots` on the Quality Plan (10 checkpoints `cp_00_plan` …
   `cp_final_submit`).
2. Cold start spawns parallel **Reviewer** audits + a **Trainer** runner-verify
   + a **Data Worker** to build a baseline `train.jsonl`.
3. **Reviewer** profiles LR sweep → **Planner** proposes a baseline recipe
   citing the profile_run / data_gap → **Trainer** executes via
   `launch_training` (platform `StageRegistry`) → **Reviewer** scores eval and
   writes a fresh `data_gap`.
4. Each worker tags evidence `checkpoint:<slot_id>` so the Quality Plan fold
   can count it. The **Reviewer** separately posts
   `checkpoint_review_suggest(verdict)` — slots only advance after that
   verdict; in manual mode a human clicks Sign.

## Where the design lives in the repo

- Subagent frontmatter + tool allowlists: [.claude/agents/nemo_mas_*.md](.claude/agents/)
- Full per-role protocol prompts: [seed_workspaces/nemo_mas_reasoner/prompts/](../../seed_workspaces/nemo_mas_reasoner/prompts/)
  - [system.md](../../seed_workspaces/nemo_mas_reasoner/prompts/system.md) (orchestrator)
  - [planner.md](../../seed_workspaces/nemo_mas_reasoner/prompts/planner.md)
  - [data_worker.md](../../seed_workspaces/nemo_mas_reasoner/prompts/data_worker.md)
  - [trainer.md](../../seed_workspaces/nemo_mas_reasoner/prompts/trainer.md)
  - [reviewer.md](../../seed_workspaces/nemo_mas_reasoner/prompts/reviewer.md)
- Quality Plan spec (the 10 slots + evidence contract): [.claude/quality_plan.md](quality_plan.md)
- Workspace contract + evolvable layers: [seed_workspaces/nemo_mas_reasoner/manifest.yaml](../../seed_workspaces/nemo_mas_reasoner/manifest.yaml)
- Training stage registry (where Trainer dispatches): `agent_evolve/model/runners/stages/*.py`

---

# Response to the "goalkeeper / critic / executor" suggestion

> 确实现在很多 agent 是在优化"把事做完"而不是"把事做好"。我建议你从
> 组织设计角度试，而不是只从 prompt 或单个 agent 能力入手。比如一个
> agent 当 goalkeeper 持续追问"这一步有没有提高最终分数"，一个 agent
> 当 critic 专门判断方案为什么只是 70 分、怎么到 80 分，执行 agent
> 负责落地。

**Totally agree with the framing.** The "optimize for done, not for good"
failure mode is exactly what we hit in earlier prototypes where a single
agent both proposed and executed — it rewarded itself for shipping records,
not for moving the metric. The current MAS already pulls apart **propose vs
execute vs audit**, but the "goalkeeper" and "critic" roles you describe are
genuinely missing as first-class agents; they're smeared across the
Orchestrator and Reviewer today, which blunts both.

## Where your idea maps onto what already exists

| Your role     | Closest current role | How it shows up today                                           | What's missing |
|---------------|----------------------|------------------------------------------------------------------|----------------|
| **Goalkeeper** | Orchestrator        | Reads `eval_report` trend before spawning, can halt the cycle if two consecutive evals show no improvement | Never asks *this specific action* — it asks *this cycle*. No per-step "does this step raise kaggle score?" gate. |
| **Critic**     | Reviewer            | Posts `ready_to_sign` / `reject` verdicts on evidence, writes `error_pattern` + `data_gap` | Judges *whether evidence is valid*, not *why the recipe only scored 70 and how to get to 80*. That analysis is squeezed into the Planner, which also has to propose — so the critique gets softened. |
| **Executor**   | Trainer + Data Worker | Already pure executors, cannot write recipes                     | ✅ This split already works well. |

## Concrete proposal for this repo

Three incremental additions, all fit the existing workspace contract without
changing the training stack:

1. **Promote "goalkeeper" to an explicit sub-role of the Orchestrator.**
   After every worker returns, the Orchestrator runs a cheap check: for the
   newest record with metrics, compare the primary metric delta vs. the
   last 3 `cv_result`s. If the delta is < seed-noise, it writes
   (or delegates a reviewer to write) a new kind `score_delta_note` and
   tags the cycle with `stalled=true`. The Planner has to cite that note
   before spending another recipe proposal — forces the "is this making
   things better?" question into the DAG rather than leaving it implicit.

2. **Split Reviewer into `auditor` (evidence validity) and `critic`
   (gap-to-frontier analysis).** Same memory, different prompts:
   - *Auditor* keeps today's QA-officer hat (`checkpoint_review_suggest`,
     `data_audit_finding`).
   - *Critic* takes only the top `cv_result` and writes a new kind
     `score_ceiling_analysis`: "we're at 0.68, SOTA-like behavior would
     be 0.80, the highest-leverage missing capability is X because error
     bucket Y accounts for 60% of the gap". Planner must ref this when
     proposing the next recipe. This is exactly the "为什么只是 70 分、
     怎么到 80 分" agent you described — and separating it from the
     auditor prevents "evidence looks healthy ✓" from masquerading as
     "recipe is good enough ✓".

3. **Goalkeeper veto on sign-off.** Today a slot closes when the reviewer
   posts `ready_to_sign` and evidence is complete. Add a goalkeeper check
   on cp_06/cp_07/cp_09: the slot cannot move to `signed` unless the
   latest `score_delta_note` says "+Δ vs. prior promoted recipe" OR a
   `score_ceiling_analysis` justifies signing despite flat score (e.g.,
   platform capability unlock). Prevents "checkpoint theater" where we
   keep signing slots while the metric flatlines.

The organizational-design lever you're pointing at is real: separating
*"this is correct"* (auditor), *"this is good enough vs. the frontier"*
(critic), and *"this step moves the scoreboard"* (goalkeeper) is exactly
how you stop the system from optimizing for turn-completion. The cost is
one more role prompt and two new memory kinds — cheap — and it lines up
with the existing refs-DAG so the trace viewer renders it for free.

Honest caveat: adding roles also adds coordination overhead and token
cost. I'd land it as an A/B — run the current 5-role team and the
7-role team on the same starting checkpoint for one cycle and compare
the delta-per-GPU-hour, not just the absolute score. If the critic/
goalkeeper roles aren't adding measurable signal beyond what Planner +
Reviewer already produce, we keep the lean version.
