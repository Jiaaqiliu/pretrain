# Skill: scaffold_sft_runner

When to use: `check_pipeline_coverage` reports the SFT stage is
not covered by the current `runner/` (cold start, OR a recipe
specifies a stage variant not yet supported).

## Inputs

- The active backend declared in `manifest.yaml::defaults::backend`
  and detailed in `backend/<backend>.yaml`.
- The active recipe (`data/recipes/default.yaml` and
  `train/pipeline.yaml`).

## Procedure

1. `read_file("backend/<backend>.yaml")` — get GPU layout, distributed
   library (FSDP / DeepSpeed / vanilla), launcher (torchrun /
   accelerate / k8s job).
2. `read_file("train/pipeline.yaml")` — confirm SFT stage is
   declared, what it expects to consume (data path, base model,
   LoRA config), what it produces (checkpoint path).
3. `read_file("model/base.yaml")` and `read_file("model/adapter.yaml")`
   to know the model + adapter shapes.
4. `read_file("train/optimizer.yaml")`, `train/loss.yaml`,
   `train/batching.yaml` — these are the inputs the runner reads at
   launch time. Do NOT inline their values; the runner should read
   the YAMLs so Theorist's diffs land without rescaffolding.
5. `scaffold_runner(stage="sft", template=<chosen template>)` —
   produce the file at `runner/sft_runner.py`.
6. Smoke test: launch a 5-step run on a 100-row subset of the
   current `data/final/train.jsonl` (or `eval/local_holdout_small.jsonl`
   for cold start). Confirm: process starts, model loads, one
   forward+backward succeeds, checkpoint can be written.
7. Update the runner_capability record:

   ```yaml
   kind: runner_capability
   title: "Runner: SFT scaffolded for <backend>"
   body: |
     Backend:        <backend>
     File:           runner/sft_runner.py
     Stages covered: [sft]
     Inputs read:
       - data path (CLI arg, default data/final/train.jsonl)
       - train/optimizer.yaml
       - train/loss.yaml
       - train/batching.yaml
       - model/base.yaml
       - model/adapter.yaml
     Outputs:
       - checkpoint dir (CLI arg)
       - metric.json beside ckpt
     Smoke test (5 steps, 100 rows): <PASS / FAIL with notes>
     Distribution lib: <FSDP | DeepSpeed | vanilla>
     Launcher:        <torchrun | accelerate | k8s>
   tags: ["sft", <backend>]
   refs: []
   ```

## Template choice (defaults)

| Backend | Template |
|---|---|
| `h200_single_node` | `templates/sft_torchrun_lora.py` |
| `k8s_h200` | `templates/sft_k8s_lora.py` |
| `h200_multi_node` | `templates/sft_fsdp_lora.py` (NYI — write `failed_attempt`) |

Templates live in `runner/templates/` (committed). When you
scaffold, copy and customize — don't edit the template.

## Hard rules

- The runner MUST read all evolvable YAML at launch time, not at
  scaffold time. (Otherwise Theorist's recipe diffs require
  re-scaffolding.)
- The runner MUST write `metric.json` next to the checkpoint with
  fields: `final_train_loss`, `final_step`, `wallclock_seconds`,
  `peak_gpu_memory_gb`. This is what
  `read_checkpoint_metric` reads.
- The runner MUST exit non-zero if any of: NaN encountered, loss
  explodes (per the divergence rule in engineer.md), CUDA OOM.

## Anti-patterns

- Do NOT inline LR or any optimizer hyperparameter into the
  scaffolded runner.
- Do NOT skip the smoke test. A runner that "looks right" but
  hasn't been smoke-tested will burn a full training_run.
- Do NOT scaffold a runner without recording `runner_capability`.
  Future Engineer spawns rely on that record to know what's
  covered.
