#!/usr/bin/env bash
# 1-cycle teacher-distillation → SFT → eval on the Kaggle Nemotron-Reasoning benchmark.
#
# Stage breakdown (one workspace fork, three pipeline stages):
#   1. synth_generate: Nemotron-Super-120B-FP8 TP=4 on GPUs 0-3 samples 500
#      prompts (250 cipher + 250 bits) from train_local.csv, filters by
#      correct + boxed + min_tokens>=2500 + student_len<=8192. Writes
#      data/synth/teacher_traces.jsonl and appends it to data/sources.yaml.
#   2. sft_warmup: rank-16 LoRA training on the mix of short_correct.jsonl
#      (476 rows) + teacher_traces.jsonl. max_steps=8, grad_accum=32, lr=5e-5.
#   3. eval: vLLM + LoRA on 951-row Kaggle dev, verbatim boxed-EM metric.
#
# GPU usage: needs 4 GPUs for the synth stage. SFT + eval then run on GPU 0
# after the teacher is torn down (synth_worker calls torch.cuda.empty_cache).
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

# Parent keeps ONE GPU visible so HF Trainer doesn't auto-DataParallel the
# 30B model across 4 devices (which OOMs because LoRA isn't sharded). The
# teacher subprocess overrides its own CUDA_VISIBLE_DEVICES via AE_SYNTH_GPUS
# so it still gets the full TP=4 quota.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  AE_SYNTH_GPUS=${AE_SYNTH_GPUS:-0,1,2,3} \
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
