# AutoPretrain Experiment Progress

## Current Status: MVP 3-Trial Ready to Submit

---

## Timeline

### 2026-05-24: Setup Complete

**Completed:**
- [x] Algorithm implementation (MCGS + 6 mutation strategies + reward + eval harness)
- [x] Literature survey (7 papers + 5 tech reports with exact data ratios)
- [x] Proxy model upgraded: 190M → OLMo2-1B (1.6B params, r=0.956 transfer to 7B)
- [x] Data verified on FSx: 4 domains, ~94B tokens total available
- [x] Code deployed to cluster FSx (no git push needed)
- [x] Full dry-run passed: all configs construct correctly on cluster
- [x] K8s YAML ready with correct naming prefix (luhanqin-)

**Data on FSx (`/fsx/dev/jiaqi/data/olmo-pretrain/`):**
| Domain | Path | Shards | Tokens |
|--------|------|--------|--------|
| web | dclm_web | 3759 | 38.5B |
| code | code | 967 | 9.7B |
| math | math | 350 | 7.0B |
| academic | fineweb_edu | 3759 | 38.5B |

---

## MVP Experiment: 3-Trial Validation

### Goal
Determine if data mixture composition has measurable impact on val loss
at 1B scale in 5000 steps. If yes → full MCGS search is justified.

### Trials

| Trial | Mix | web | code | math | academic | Status |
|-------|-----|-----|------|------|----------|--------|
| 1 | llama3 | 52% | 18% | 26% | 4% | READY |
| 2 | reasoning_heavy | 25% | 30% | 30% | 15% | READY |
| 3 | uniform | 25% | 25% | 25% | 25% | READY |

### Config
- Model: OLMo2-1B (d_model=2048, n_layers=18, 1.6B params)
- Steps: 5000
- Batch: 128 seq × 4096 tokens = 524K tokens/step
- Total tokens per trial: ~2.6B
- LR: 3e-4, warmup=500, cosine decay
- Hardware: 1 node × 8 H200 (FSDP, bf16)
- Estimated time: ~1.5 hours per trial

### Submit Command
```bash
kubectl apply -f /fsx/dev/jiaqi/A-EVOLVE-V2/experiments/autopretrain/k8s_mvp_3trial.yaml \
  -n default --context arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm
```

### Monitor
```bash
kubectl get jobs | grep luhanqin-autopretrain
kubectl logs job/luhanqin-autopretrain-llama3 --follow
```

### Success Criteria
- All 3 trials complete without crash
- Val loss differs by > 0.05 between trials
- If reasoning_heavy < llama3 < uniform → proceed to Phase 1 full search

### Results (TBD)

| Trial | Final Loss | Perplexity | ARC-Easy | PIQA | Notes |
|-------|-----------|------------|----------|------|-------|
| llama3 | | | | | |
| reasoning_heavy | | | | | |
| uniform | | | | | |

---

## Next Steps (after MVP)

### If MVP succeeds (mixtures show clear signal):
1. **Phase 1**: 30-cycle MCGS search at 1B (360 GPU-hours)
2. **Phase 2**: Transfer top-3 to 3B (500 GPU-hours)
3. **Phase 3**: Full 60B training + Kaggle (3000 GPU-hours)

### If MVP fails (no signal at 5000 steps):
- Increase to 10000 steps per trial (double compute)
- Or switch to 3B proxy directly (higher signal, more expensive)
- Or add per-domain val loss to detect finer differences

---

## Reasoning Evaluation Plan

### Fast eval (after each trial, ~10 min):
```bash
# Run inside GPU pod after training completes
pip install lm-eval
python -m lm_eval \
  --model hf \
  --model_args pretrained=/fsx/dev/jiaqi/checkpoints/autopretrain-mvp/<trial> \
  --tasks arc_easy,piqa,hellaswag \
  --batch_size 16
```

### Full eval (Phase 3 only):
- LoRA SFT on NeMo Reasoning data → Kaggle submission
- Full benchmark suite: MMLU, GSM8K, HumanEval, ARC, PIQA

---

## Key Decisions Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-05-24 | Proxy 190M → 1B | DCLM: r=0.956 vs 0.838 transfer; 1B still fast enough |
| 2026-05-24 | Drop books domain | Empty on FSx; 4 domains sufficient |
| 2026-05-24 | Start from Llama-3 mix | Strongest reasoning baseline (25% math) |
| 2026-05-24 | Code on FSx not remote | Avoid pushing experimental branch; kubectl cp instead |
