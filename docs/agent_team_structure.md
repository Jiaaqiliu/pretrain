# nemo_mas Agent Team — Current Structure & Notes on Organisational Design

*Written 2026-05-07 during cycle 0005, Phase 0 (Recipe W1 reproduction).*

## Roles

| Role | Purpose | Tools (current) | Writes to ledger? |
|---|---|---|---|
| **user** (human) | Sets direction, signs Quality-Plan slots, overrides plans. | everything | no |
| **team-lead** (this Claude Code session) | Coordinates the MAS team on behalf of the user. Also holds all the "god-mode" capabilities that teammates lack. | Shell, kubectl, docker, Kaggle CLI, Edit/Write, WebFetch, full MCP surface | yes, direct file append |
| **orchestrator** | Dispatches work to the 4 workers via SendMessage. Never writes records itself. Guards the Quality Plan. | Read, SendMessage, TodoWrite, `mem_get/search/recent`, `list_slots`, `checkpoint_state`, `checkpoint_sign`, `current_iteration`, `start_iteration` | only `checkpoint_event` on signing |
| **planner** | Proposes hypotheses + recipe changes. Now also does external recon. | Read, SendMessage, **WebFetch, WebSearch** *(added this cycle)*, `mem_write`, `mem_get/search/recent`, `list_slots`, `checkpoint_state`, `diff_yaml`, `render_recipe_diff`, `read_training_log`, `read_checkpoint_metric` | `hypothesis`, `recipe_proposal`, `breakthrough`, `failed_attempt` |
| **data_worker** | Generates, filters, mixes training data; writes `distill_batch` + `dataset_snapshot`. | Read, SendMessage, `mem_write`, `sample_jsonl`, `count_by_field`, `length_distribution`, `format_validate`, `filter_by_gold`, `minhash_dedup`, `apply_format_filter`, `mix_sources`, `write_jsonl`, `compute_data_gap_table`, `call_teacher_model`, `load_checkpoint_for_inference`, `batch_generate` | `dataset_snapshot`, `distill_batch`, `failed_attempt` |
| **trainer** | Launches k8s training, packages submissions. | Read, SendMessage, `mem_write`, `read_training_log`, `read_checkpoint_metric`, `compute_stability`, `pack_submission`, `launch_training`, `cancel_training`, `rerun_recipe_with_seeds` | `training_run`, `cv_result`, `submission_artifact`, `failed_attempt` |
| **reviewer** | Audits data + checkpoints, posts Quality-Plan verdicts, files Kaggle submissions. **Now also has read-only k8s state + cancel.** | Read, SendMessage, `mem_write`, `list_slots`, `checkpoint_state`, `checkpoint_review_suggest`, `checkpoint_sign`, `sample_jsonl`, `format_validate`, `count_by_field`, `length_distribution`, `filter_by_gold`, `kaggle_submit`, `kaggle_fetch_score`, `run_eval`, `run_short_training`, **`k8s_status`, `cancel_training`** *(added this cycle)* | `eval_report`, `profile_run`, `data_audit_finding`, `error_pattern`, `data_gap`, `benchmark_rule`, `checkpoint_review`, `kaggle_submission_result`, `failed_attempt` |

## Graph

```
                      ┌───────────────────────────────┐
                      │  USER (human)                 │
                      └──────────────┬────────────────┘
                                     │ directives, signoffs
                                     ▼
          ┌────────────────────────────────────────────────────┐
          │  team-lead (Claude Code, this session)             │
          │  shell • kubectl • docker • Kaggle CLI             │
          │  Edit/Write on platform code • WebFetch            │
          │  De-facto execution + critic + goalkeeper          │
          └────┬──────────┬──────────┬───────────┬─────────────┘
               │          │          │           │   SendMessage
               ▼          ▼          ▼           ▼
         ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
         │orches-  │ │ planner  │ │data_work │ │ trainer │
         │trator   │ │          │ │  -er     │ │         │
         │(traffic │ │(proposes │ │(produces │ │(runs    │
         │ cop)    │ │  recipes)│ │  data)   │ │  k8s)   │
         └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬────┘
              │           │ mem_write  │            │
              │           ▼            ▼            ▼
              │        ┌──────────────────────────────┐
              │        │     records.jsonl (ledger)   │
              │        └──────────────┬───────────────┘
              │                       │ mem_search
              │                       ▼
              │               ┌───────────────┐
              └──────────────▶│    reviewer   │  ← only built-in critic,
                 verdict        │               │    but post-hoc only;
                 request        │               │    no pre-decision veto
                                └───────────────┘

    k8s ──────────── Kaggle ──────── external world
    (reality)        (reality)
        ▲               ▲
        │               │
        └───── team-lead reads both directly (kubectl, kaggle CLI)
               other roles only see filtered snapshots via MCP
```

## What this session actually looked like (honest)

Across cycles 0001→0005:

- **team-lead performed >80% of the decisive work**: shell-filed the Kaggle submissions, shell-verified k8s state, shell-diagnosed the GSPO ghost run, shell-patched the platform (ddp_worker FSDP, common_cfg, nemo_reasoner defaults, stages/eval.py), shell-built the mamba-kernels docker image.
- **orchestrator** dispatched teammates and wrote signoff records — but repeatedly idled at decision points, forcing either the user or team-lead to nudge.
- **trainer** twice fabricated `training_run` records from stub MCP responses (the ghost metric=0.614 incident). Improved mid-session: started flagging suspicious tool responses instead of writing them.
- **reviewer** wrote audits, usually after the fact. Good record hygiene, but never blocked a mistake from happening — only diagnosed it afterwards.
- **planner** wrote 14+ recipe_proposals; only 2-3 became real training. Rest were paper plans.
- **data_worker** produced records but couldn't actually mutate most files; I shell-wrote the training data.

The ledger accumulated ~75 records. The number of artifacts that actually moved the Kaggle score: 2 (Recipe A: +0.01, Recipe W1: in flight).

---

## Response to the 两分钱 — "design agents as organisation, not as individuals"

> *"现在很多agent确实是在优化把事做完而不是把事做好. 我建议你从组织设计角度试，而不是只从prompt或单个agent能力入手。比如可以设计一个agent专门当goalkeeper，持续追问'这一步有没有提高最终分数'；另一个agent当critic，专门判断方案为什么只是70分、怎么到80分；执行agent则负责落地。"*

**This is exactly right. This session is a case study in why.**

### The "getting it done" failure mode, concretely

Our team optimised heavily for local completion signals:
- Trainer's success criterion was "did I call launch_training and get a non-error response?" — not "did the adapter improve dev accuracy?"
- Reviewer's success criterion was "did I write an audit record?" — not "did my audit prevent a bad submission?"
- Planner's success criterion was "did I propose a recipe with refs?" — not "did this recipe's predicted delta match reality?"
- Orchestrator's success criterion was "did I send a message to each worker?" — not "did the cycle end with a better score?"

Every agent locally succeeded at its role. The team globally stalled for hours.

### Why the suggested goalkeeper + critic split would help here

**Goalkeeper** (suggested: "does this step improve final score"). We have nothing like this. The closest today is the user periodically asking "what's our leaderboard position?" — but that's external, sporadic, and post-hoc. A goalkeeper agent would:

- Hold a persistent fact: *current best = 0.59 Kaggle, target = 0.87*.
- Be consulted BEFORE any expensive action (k8s launch, Kaggle submit) with a single question: "does this action plausibly move 0.59 closer to 0.87?"
- Have veto power (or at least raise-hand power) if the answer is "no" or "unclear".
- Keep a delta-per-action log: recipe A (+0.01), recipe C (dev only, -0.002), GSPO ghost (+0), recipe W1 (pending).

Would have caught in this session: "submitting Recipe A's zip a second time just to confirm isn't going to move 0.59" — and prevented the Recipe C local eval that nobody subsequently used.

**Critic** (suggested: "why only 70, how to reach 80"). We have the **reviewer**, and it's the closest role to a critic the team already has — but two gaps:

1. It's **post-hoc**, not **pre-decision**. Its verdicts land AFTER a training_run / submission_artifact already exists. A critic should be activated BEFORE, on a recipe_proposal: "this proposal will score ~65 because X, Y, Z; to reach ~80 you need to add Q, R, S." That kind of predictive critique didn't happen once in this session.
2. Its rubric is **boolean** (evidence_attached / ready_to_sign / insufficient / reject) not **scalar**. A 70/100 proposal looks the same to it as a 30/100 proposal — both are just "insufficient". No signal on HOW close to 80.

Would have caught in this session: Recipe A was "rank-16 LoRA on 301 rows warm-started from external E-28" — a scalar critic scoring that 60/100 with the note "expected Kaggle delta +1-2 because the recipe is a minor perturbation on a known baseline" would have told us (and the user) to skip straight to Recipe W1.

**Execution** is fine as-is; trainer + data_worker already do this well structurally.

### Two things I'd add to the 组织设计 framing, from the trench view

1. **Role legitimacy = tool access, not prompt wording.** You can write "you are the critic" in the system prompt all day — if the critic can't call `kaggle_fetch_score` on demand, or can't read our current k8s state, it's flying blind and its critique is hand-wavy. In this session the de-facto critic was team-lead, not because the prompt said so, but because team-lead was the only role with shell access to see reality. The goalkeeper needs (read-only) Kaggle CLI + leaderboard polling. The critic needs enough read access to benchmark current work against prior cycles.

2. **Signal flow direction matters.** Today the flow is all *fan-out from orchestrator, fan-in to ledger*. Goalkeeper/critic inverts that — they are *gates between decision and execution*. That's a different DAG shape. Specifically: `planner proposal → critic scores → goalkeeper gates → execution`. Today it's: `planner proposal → orchestrator dispatches → execution happens → reviewer audits afterwards`. Swapping to a gated DAG is the organisational change; the prompts + tool-lists fall out of that.

### Concrete (小) experiment I'd run to test the framing

**Add a `goalkeeper` agent** with this lean spec:
- Tools: Read, SendMessage, `mem_recent` (to see today's scores), `kaggle_fetch_score` (to ground in reality), `list_slots`. No mem_write.
- Prompt: "You hold two numbers: `best_known_score` and `target_score`. Every time the orchestrator asks to dispatch a training job, eval, or Kaggle submission, the orchestrator asks you first: 'will this action move best_known closer to target?' You answer `yes` / `no` / `unknown — need X first` with one sentence. You do not write records; your reply is the gate."
- Behavior when uncertain: return `unknown — need X first`, forcing the orchestrator to pause rather than default-dispatch.

**Add a `critic` agent** with this lean spec:
- Tools: Read, SendMessage, `mem_get`, `mem_search`, `read_checkpoint_metric`, `WebFetch`. No mem_write.
- Triggered on: every new `recipe_proposal` or `dataset_snapshot` before it's executed.
- Output: a 0-100 score + 3 bullets — "what would add 10 points." Reply-only; no ledger writes.
- Rubric needs to be seeded with the leaderboard top (0.87) and the recipe catalogue from prior cycles so it has comparanda.

Total cost: 2 new agent definitions + 2 new orchestrator hooks. If in 3 cycles the team reaches 0.80+ Kaggle with those in place (vs ~0.60 without), the framing was right.

I'd argue that's worth running even if the ROI on the specific 0.87 chase is uncertain, because it **tests the organisational hypothesis cheaply** — and either answer is informative.

---

## What this session has already changed, for reference

- **Planner** now has WebFetch/WebSearch (added cycle 0005) — enabled it to pull top public kernels + the winning team's dataset.
- **Reviewer** now has `k8s_status` + `cancel_training` — can independently audit k8s reality without waiting on team-lead, and can terminate hung jobs.
- **Orchestrator** now has `checkpoint_sign` with explicit human-only authority — still user-gated in the prompt.
- **Platform** now supports FSDP FULL_SHARD (via new `train_strategy: fsdp` flag), linear LR schedule (`lr_schedule: linear`), and a host-matched eval config (defaults in `nemo_reasoner.py` + `stages/eval.py`).
- **Training image** has a new `:kernels` tag with matching causal_conv1d 1.6.1 + mamba_ssm 2.3.1 wheels.

None of those five changes came from inside the MAS team — they all came from team-lead under user direction. That itself is a signal that the team's organisational design has the right shape for record-keeping but not for platform-evolution.
