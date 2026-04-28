#!/usr/bin/env bash
# 1-cycle GSPO-only run on the Kaggle Nemotron-Reasoning benchmark.
#
# Stage breakdown (one workspace fork, one active pipeline stage):
#   1. rl_gspo: load base model + seed LoRA via vLLM (rollout client), sample
#      G=n_samples completions per prompt for per_domain prompts across the
#      configured domains, record per-token logprobs_old. Drop the vLLM
#      engine, load the HF + PEFT training client, group-normalize
#      advantages within (domain, pid), apply GSPO/DAPO clipped loss. Save
#      updated adapter.
#   2. eval: vLLM + LoRA on 951-row Kaggle dev, verbatim boxed-EM metric.
#
# Pre-req: model/adapter.yaml::seed_adapter_path MUST point at an existing
# SFT adapter (e.g. ../nemotron-auto-research E-28). The GSPO stage rolls out
# against this as the "old" policy. If it's unset, this run will fail loudly.
#
# GPU usage: 1 GPU is sufficient (vLLM rollout and HF training run
# sequentially — vLLM is torn down before the training client boots).
#
# Wallclock budget: ~60-90 min on 1x H200.
#
# To re-run cleanly:
#   rm -rf runs/gspo-1cycle/nemotron_reasoner runs/gspo-1cycle/nodes

set -euo pipefail

AE=/fsx/zzsamshi/a-evolve
NEMOTRON_VENV=/fsx/zzsamshi/nemotron-auto-research/.venv

cd "$AE"
mkdir -p runs/gspo-1cycle/logs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  PYTHONPATH="$AE" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  WANDB_DISABLED=true \
  VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 \
  VLLM_USE_FLASHINFER_MOE_FP8=0 \
  VLLM_USE_FLASHINFER_MOE_FP4=0 \
  VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  "$NEMOTRON_VENV/bin/python" \
  "$AE/examples/nemo_reasoning_example/drive_gspo_1cycle.py" \
  > runs/gspo-1cycle/logs/run.log 2>&1 &

echo "pid=$!"
echo "tail: tail -f $AE/runs/gspo-1cycle/logs/run.log"
