# AutoPretrain

**World-class autonomous LLM pretraining orchestration framework.**

AutoPretrain is a self-healing, self-evolving training agent that runs for days without human intervention. It automatically diagnoses failures, selects recovery strategies, and learns from outcomes to handle novel issues progressively better.

## Key Features

- **Formal State Machine** — deterministic job lifecycle (CREATED → PENDING → RUNNING → COMPLETED/DIAGNOSING → RECOVERING)
- **Multi-Layer Failure Diagnosis** — heartbeat + log regex + K8s events + correlation analysis + self-learning
- **Adaptive Recovery** — escalating fixes (e.g., OOM: reduce batch → grad ckpt → aggressive reduction → alert)
- **Self-Evolving Intelligence** — learns from successful fixes; second encounter = instant match
- **Multi-Scale** — supports OLMo2 190M to 32B with automatic resource allocation
- **Full Parameter Search** — data mixtures + learning rate + batch size + warmup
- **Real-Time Monitoring** — loss spike, gradient explosion, NaN, throughput drop, divergence, plateau detection
- **Complete Audit Trail** — every decision logged with evidence and reasoning

## Quick Start

```bash
# On cluster CPU pod:
cd /fsx/dev/jiaqi/A-EVOLVE-V2

# Run 3-trial data mix comparison
python -m autopretrain run \
  --trials llama3,reasoning_heavy,uniform \
  --model olmo2_1B \
  --steps 5000

# Check status
python -m autopretrain status

# View detailed event log
python -m autopretrain events --tail 20
```

## Architecture

```
autopretrain/
├── core/              # Types + Protocol interfaces
├── orchestrator/      # State machine + async event loop
├── resilience/        # Diagnosis + recovery + self-learning
├── monitor/           # Statistical anomaly detection
├── compute/           # K8s backend (async kubectl)
├── engine/            # OLMo-core script/manifest generation
├── search/            # Data mix + hyperparameter mutation
├── store/             # JSONL audit trail
├── deploy/            # K8s deployment manifests
└── cli.py             # Command-line interface
```

## Design Principles

| # | Principle | Source |
|---|-----------|--------|
| 1 | Training script fast-fail + Agent recovers | Composer, torchtitan |
| 2 | Formal state machine (not if/else) | Ray Train |
| 3 | Heartbeat + dynamic timeout | NeMo, Determined |
| 4 | Scaling event ≠ Failure event | DeepSpeed |
| 5 | Atomic checkpoint (metadata-last) | OLMo-core |
| 6 | Problem node exclusion | Determined AI |
| 7 | Event-Action mapping | Volcano |
| 8 | Confidence-gated auto-action | AutoTrain |
| 9 | Budget-aware retry | All |
| 10 | Complete audit trail | All |

## Supported Models

| Model | Params | GPUs | Factory |
|-------|--------|------|---------|
| OLMo2-190M | 190M | 1 | `olmo2_190M` |
| OLMo2-1B | 1.6B | 8 | `olmo2_1B` |
| OLMo2-3B | 3.3B | 8 | `olmo2_3B` |
| OLMo2-7B | 7B | 8 | `olmo2_7B` |
| OLMo2-13B | 13B | 16 | `olmo2_13B` |
| OLMo2-32B | 32B | 32 | `olmo2_32B` |

## Self-Healing Flow

```
Job Fails → Gather Signals (logs, events, heartbeat)
         → Multi-Layer Diagnosis (regex → correlation → self-learn)
         → Select Recovery (event-action table, escalating)
         → Safety Validation (confidence > 0.7, budget check)
         → Apply Fix → Resubmit
         → Observe Outcome → Update Knowledge Base
```

## Testing

```bash
PYTHONPATH=. python -m autopretrain.tests.test_integration
# 19 tests covering all core flows
```
