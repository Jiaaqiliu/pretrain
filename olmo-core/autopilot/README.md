# AutoPilot 🚀

**The world's first autonomous LLM training agent.**

AutoPilot is an open-source system that independently manages the full lifecycle of large language model training — from pre-training through mid-training to post-training — with minimal human intervention.

## Key Capabilities

- **Natural language control**: Configure training through conversational commands
- **Autonomous HP optimization**: muTransfer-based search at ~1% compute cost
- **Smart data management**: Automatic mixture optimization (DoReMi/RegMix)
- **Real-time monitoring**: Anomaly detection, loss prediction, early stopping
- **Multi-experiment management**: Parallel experiments with ASHA pruning
- **Self-healing**: Automatic failure detection and checkpoint recovery
- **Long-running operation**: Persistent state, runs autonomously for days/weeks
- **Multi-backend**: Supports Beaker, SLURM, Kubernetes clusters

## Quick Start

```bash
# Install
cd autopilot
pip install -e '.[all]'

# Start a training campaign with natural language
autopilot train --model-size large --data-domains web code math \
    --num-nodes 8 --backend slurm --autonomy semi

# Or use the interactive mode
autopilot interactive
```

## How It Works

```
User: "Train a 7B model on web and code data with 8 nodes"
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Phase 1: Proxy Search (muTransfer)                 │
│  → 20 parallel trials on 190M proxy model           │
│  → Find optimal LR, weight decay, scheduler         │
│  → Cost: ~1% of target training                     │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2: Data Mixture Optimization                 │
│  → DoReMi/RegMix on proxy model                    │
│  → Find optimal domain weights                      │
│  → Cost: ~5% of target training                     │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3: Validation Run                            │
│  → Transfer HPs to 1B model                        │
│  → Verify scaling behavior                          │
│  → Confirm configuration works                      │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│  Phase 4: Full-Scale Training                       │
│  → 7B model with optimized configuration            │
│  → Continuous monitoring + anomaly detection         │
│  → Dynamic data mixture adjustment                  │
│  → Automatic checkpoint recovery on failures        │
└─────────────────────────────────────────────────────┘
```

## Architecture

```
autopilot/
├── agent/              # The brain — planning, analysis, decisions
│   ├── orchestrator    # Central coordinator
│   ├── planner         # Multi-phase campaign planning
│   ├── analyzer        # Experiment analysis and insights
│   ├── decision        # Autonomous decision engine
│   ├── natural_language # NL command parsing
│   ├── data_manager    # Data discovery and management
│   └── environment     # Environment discovery and setup
├── experiment/         # The hands — job management
│   ├── config_builder  # Dynamic config generation
│   ├── launcher        # Job submission
│   └── monitor         # Real-time monitoring
├── optimization/       # The intelligence — optimization algorithms
│   ├── hpo            # Hyperparameter optimization (Optuna)
│   ├── mu_transfer    # muTransfer HP scaling
│   ├── data_mixing    # DoReMi/RegMix mixture optimization
│   └── early_stopping # ASHA + predictive stopping
├── monitoring/         # The eyes — metrics and anomalies
│   ├── metrics        # Metrics collection and windowing
│   ├── anomaly        # Multi-signal anomaly detection
│   └── prediction     # Loss curve extrapolation
├── backends/           # The feet — compute infrastructure
│   ├── beaker         # Beaker cluster support
│   ├── slurm          # SLURM cluster support
│   └── kubernetes     # K8s support (planned)
└── ui/                 # The face — user interfaces
    └── cli            # Rich CLI with dashboard
```

## Autonomy Levels

| Level | Behavior |
|-------|----------|
| **Full** | Agent makes all decisions automatically |
| **Semi** | Low-risk actions auto-execute; high-risk require confirmation |
| **Advisory** | Agent only suggests; user executes |

## Built-in Data Sources

AutoPilot knows about major open-source training datasets and can:
- Suggest sources to fill data gaps
- Generate download/tokenization pipelines
- Optimize mixture ratios automatically

Supported: FineWeb, Dolma, RedPajama, StarCoder, The Stack, OpenWebMath, peS2o, Wikipedia, SlimPajama, DCLM, and more.

## Configuration

### Training Targets (what to train)

```yaml
# configs/targets/7b_pretrain.yaml
model_size: large
phase: pretrain
target_tokens: 2_000_000_000_000
compute_budget:
  num_nodes: 8
  gpus_per_node: 8
  gpu_type: A100-80GB
data_domains: [web, code, books, academic, math]
```

### Strategies (how to search)

```yaml
# configs/strategies/aggressive_search.yaml
proxy_search:
  n_trials: 30
  width_divisor: 4
search_space:
  learning_rate: {type: log_float, low: 1e-5, high: 1e-2}
  weight_decay: {type: float, low: 0.0, high: 0.3}
early_stopping:
  patience: 2000
  asha_reduction_factor: 3
data_mixing:
  strategy: doremi
```

## Research Foundation

AutoPilot is built on solid research:

- **muTransfer** (Yang et al. 2022) — Zero-shot HP transfer across scales
- **DoReMi** (Xie et al. 2023) — Domain reweighting with minimax optimization
- **Chinchilla** (Hoffmann et al. 2022) — Compute-optimal training allocation
- **ASHA** — Asynchronous successive halving for experiment pruning
- **Wortsman et al. 2023** — Small-scale proxies for training instabilities

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- OLMo-core (training engine)
- Optuna (hyperparameter optimization)
- Access to a GPU cluster (Beaker, SLURM, or K8s)

## License

Apache 2.0
