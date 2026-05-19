# legacy — solver-as-hint + LLM-teacher pipelines

Five-stage generation pipelines that combine a deterministic Python
solver (witness extraction) with a 122B Nemotron teacher LLM that writes
natural-language CoT around the solver's hint.

```
stage_1_witness_search   (Python solver finds a rule fitting examples + Kaggle answer)
stage_2_teacher_distill  (122B teacher writes CoT around the hint)
stage_3_self_distill     (optional self-distill)
stage_4_opus_supervision (Opus 4.6 audits CoT quality)
stage_5_curate           (filter + dedup + format align)
```

These pipelines are **still active** — the planner / data_worker
skills (`dw-pipeline-launch`, `trainer-ablation-launch`,
`nemo_mas_*`) all drive them. The directory is named `legacy/` to
distinguish them from the newer programmatic pipeline at
[../cot_rules/](../cot_rules/), which is LLM-free.

## Files

| Path                          | Role                                          |
|-------------------------------|-----------------------------------------------|
| `shared/run_pipeline.py`      | domain-agnostic 5-stage driver                |
| `shared/stages/`              | per-stage modules (witness / teacher / opus / curate / self) |
| `shared/hint_providers/`      | per-domain solver wrappers (bits, equations)  |
| `shared/extract_subset.py`    | per-category ablation arms                    |
| `shared/leaderboard.py`       | per-category data-efficiency view             |
| `bits/pipeline.yaml`          | bits-specific 5-stage config                  |
| `bits/prompt_templates.yaml`  | bits-specific prompts                         |
| `equations/pipeline.yaml`     | equations-specific 5-stage config             |
| `equations/prompt_templates.yaml` | equations-specific prompts                |

Domain solvers are at [../../reasoners/](../../reasoners/) and the family-F
verifiers at [../../verifiers/](../../verifiers/) — both shared with
`cot_rules/`.

## Running

```bash
python -m agent_evolve.model.data.pipelines.legacy.shared.run_pipeline \
    --config   agent_evolve/model/data/pipelines/legacy/bits/pipeline.yaml \
    --templates agent_evolve/model/data/pipelines/legacy/bits/prompt_templates.yaml
```

For per-category ablations and per-batch leaderboard:

```bash
python -m agent_evolve.model.data.pipelines.legacy.shared.extract_subset ...
python -m agent_evolve.model.data.pipelines.legacy.shared.leaderboard
```

See [.claude/skills/dw-pipeline-launch/SKILL.md](../../../../.claude/skills/dw-pipeline-launch/SKILL.md)
for the full launch protocol.
