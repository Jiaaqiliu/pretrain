#!/usr/bin/env bash
# 1-cycle teacher-distillation → SFT → eval on the Kaggle Nemotron-Reasoning benchmark.
#
# Stage breakdown (one workspace fork, three pipeline stages):
#   1. synth_generate: Nemotron-Super-120B-FP8 TP=8 on all GPUs samples 500
#      prompts (250 cipher + 250 bits) from train_local.csv, filters by
#      correct + boxed + min_tokens>=2500 + student_len<=8192. Writes
#      data/synth/teacher_traces.jsonl and appends it to data/sources.yaml.
#   2. sft_warmup: rank-16 LoRA training via HFTrainingClient with
#      device_map="auto" — the 30B MoE shards across all 8 GPUs.
#      max_steps=8, grad_accum=32, lr=5e-5.
#   3. eval: vLLM TP=8 + LoRA on 951-row Kaggle dev, verbatim boxed-EM metric.
#
# GPU usage: all 8 GPUs are used end-to-end. Teacher subprocess tears down
# its CUDA state on exit so SFT gets a clean box.
#
# Wallclock budget: ~75-95 min. Trial budget is 2h hard cap.
#
# To re-run cleanly:
#   rm -rf runs/teacher-distill-1cycle/nemotron_reasoner runs/teacher-distill-1cycle/nodes

set -euo pipefail

AE=/fsx/zzsamshi/a-evolve
NEMOTRON_VENV=/fsx/zzsamshi/nemotron-auto-research/.venv

cd "$AE"
mkdir -p runs/teacher-distill-1cycle/logs

# Parent keeps all 8 GPUs visible. HFTrainingClient uses device_map="auto"
# (configured via model/adapter.yaml) to shard the 30B model across them.
# The teacher synth subprocess inherits CUDA_VISIBLE_DEVICES via AE_SYNTH_GPUS
# and runs TP=8 to match.
unset CUDA_VISIBLE_DEVICES
AE_SYNTH_GPUS=${AE_SYNTH_GPUS:-0,1,2,3,4,5,6,7} \
  PYTHONPATH="$AE" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  WANDB_DISABLED=true \
  VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 \
  VLLM_USE_FLASHINFER_MOE_FP8=0 \
  VLLM_USE_FLASHINFER_MOE_FP4=0 \
  VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  "$NEMOTRON_VENV/bin/python" \
  "$AE/examples/nemo_reasoning_example/drive_teacher_distill_1cycle.py" \
  > runs/teacher-distill-1cycle/logs/run.log 2>&1 &

echo "pid=$!"
echo "tail: tail -f $AE/runs/teacher-distill-1cycle/logs/run.log"
