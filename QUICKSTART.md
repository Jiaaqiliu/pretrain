# A-Evolve Quick Start Guide

A practical guide to using the `agent-evolve` package — from installation through running your first evolution loop.

## Table of Contents

- [Installation](#installation)
- [Core Concepts (2 minutes)](#core-concepts)
- [Tutorial 1: Run a Built-in Benchmark](#tutorial-1-run-a-built-in-benchmark)
- [Tutorial 2: Evolve an Agent](#tutorial-2-evolve-an-agent)
- [Tutorial 3: Bring Your Own Agent](#tutorial-3-bring-your-own-agent)
- [Tutorial 4: Write a Custom Evolution Algorithm](#tutorial-4-write-a-custom-evolution-algorithm)
- [Benchmark-Specific Guides](#benchmark-specific-guides)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.11+
- Docker (for SWE-bench, MCP-Atlas, Terminal-Bench)
- AWS credentials configured (for Bedrock model access)

### Install with uv (recommended)

```bash
# Create a virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install from source
git clone https://github.com/A-EVO-Lab/a-evolve.git && cd a-evolve
uv pip install -e ".[all,dev]"
```

### Install specific extras only

```bash
uv pip install -e ".[anthropic]"     # Claude support only
uv pip install -e ".[mcp]"           # MCP-Atlas benchmark
uv pip install -e ".[swe]"           # SWE-bench benchmark
```

### Verify installation

```python
import agent_evolve as ae
print(ae.__version__)        # 0.1.0
print(ae.Evolver)            # <class 'agent_evolve.harness.api.Evolver'>
print(ae.BaseAgent)          # <class 'agent_evolve.harness.protocol.base_agent.BaseAgent'>
```

---

## Core Concepts

A-Evolve has **three pluggable components** connected by a **file system contract**:

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│    Agent     │     │    Workspace    │     │  Evolution Algo  │
│  (solves)    │────>│   (FS files)    │<────│   (mutates)      │
└─────────────┘     └─────────────────┘     └──────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Benchmark  │
                    │ (evaluates) │
                    └─────────────┘
```

### The Workspace (file system contract)

All evolvable state lives as plain files in a directory:

```
my_agent/
├── manifest.yaml          # Agent identity + what's evolvable
├── prompts/system.md      # System prompt
├── skills/                # SKILL.md files (dynamic procedures)
├── tools/                 # Tool definitions
└── memory/                # Episodic memory (JSONL)
```

The evolution engine **writes** to these files. The agent **reads** from them. They never talk directly.

### The Evolution Loop

```
Solve → Observe → Evolve → Gate → Reload → (repeat)
```

1. **Solve**: Agent processes a batch of tasks
2. **Observe**: Benchmark evaluates, feedback is logged
3. **Evolve**: Algorithm analyzes failures, mutates workspace files
4. **Gate**: Validate on holdout tasks — rollback bad mutations
5. **Reload**: Agent reads updated files, ready for next cycle

Every mutation is git-tagged (`evo-1`, `evo-2`, ...) for full reproducibility.

---

## Tutorial 1: Run a Built-in Benchmark

The simplest way to test — run the baseline (no evolution) on MCP-Atlas.

### Step 1: Pull the Docker image

```bash
docker pull ghcr.io/scaleapi/mcp-atlas:latest
```

### Step 2: Set up API keys (optional)

Create a `.env` file in the repo root. Without API keys, only keyless MCP servers are available (~40% of tasks):

```bash
# .env (optional — enables more tasks)
BRAVE_API_KEY=brv-xxxxxxxxxxxx
EXA_API_KEY=xxxxxxxxxxxx
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
```

### Step 3: Run baseline

```bash
python examples/mcp_examples/adaptive_evolve_baseline.py \
  --solver-model us.anthropic.claude-opus-4-6-v1 \
  --judge-model us.anthropic.claude-sonnet-4-20250514-v1:0 \
  --region us-west-2 \
  --docker-image ghcr.io/scaleapi/mcp-atlas:latest \
  --limit 5 \
  --batch-size 5 \
  --output-dir results/mcp_baseline
```

### Step 4: Check results

```bash
cat results/mcp_baseline/summary.csv
# task_id, result, score, elapsed_s, output_len, detail
```

Each task also produces:
- `output_<task_id>.txt` — Agent's final answer
- `conversation_<task_id>.json` — Full tool-call trajectory

---

## Tutorial 2: Evolve an Agent

Now add the evolution loop — solve a batch, evolve, repeat.

### MCP-Atlas with evolution

```bash
python examples/mcp_examples/adaptive_evolve_all.py \
  --solver-model us.anthropic.claude-opus-4-6-v1 \
  --evolver-model us.anthropic.claude-sonnet-4-20250514-v1:0 \
  --judge-model us.anthropic.claude-sonnet-4-20250514-v1:0 \
  --region us-west-2 \
  --docker-image ghcr.io/scaleapi/mcp-atlas:latest \
  --limit 30 \
  --batch-size 10 \
  --seed-workspace seed_workspaces/mcp \
  --work-dir ./evolution_workdir/mcp \
  --output-dir results/mcp_evolved
```

### What happens

After each batch, the `AdaptiveEvolveEngine`:
1. Analyzes per-claim feedback from the judge
2. Identifies failure patterns by claim type
3. Generates targeted skills (e.g., `entity-validation`, `calculate-handler`)
4. Applies auto-fixes to the system prompt
5. Git-commits and tags the mutation

### Inspect the evolved workspace

```bash
# See evolution history
cd evolution_workdir/mcp
git log --oneline --tags

# Compare before vs after
git diff evo-0..evo-3 -- prompts/ skills/ memory/

# List evolved skills
ls skills/
# calculate-handler/  entity-validation/  multi-requirement-handler/
```

### SWE-bench with evolution

```bash
python examples/swe_examples/evolve_sequential.py \
  --model-id us.anthropic.claude-opus-4-6-v1 \
  --region us-west-2 \
  --batch-size 5 \
  --parallel 3 \
  --feedback minimal \
  --seed-workspace seed_workspaces/swe \
  --output-dir results/swe_evolved \
  --limit 20
```

### Terminal-Bench with evolution

```bash
BYPASS_TOOL_CONSENT=true python examples/tb_examples/batch_evolve_terminal.py \
  --solver react \
  --workers 2 \
  --batch-size 5 \
  --seed-workspace seed_workspaces/terminal \
  --work-dir ./evolution_workdir/terminal \
  --log-dir results/tb2_evolved \
  --output results/tb2_evolved/results.jsonl \
  --errors results/tb2_evolved/errors.jsonl
```

---

## Tutorial 3: Bring Your Own Agent

Make any agent evolvable in 3 steps.

### Step 1: Create a workspace directory

```
my_agent/
├── manifest.yaml
├── prompts/
│   └── system.md
├── skills/
├── tools/
└── memory/
```

**manifest.yaml:**
```yaml
name: my-custom-agent
version: 0.1.0
contract_version: "1.0"

agent:
  type: reference
  entrypoint: my_module.MyAgent

evolvable_layers:
  - prompts
  - skills
  - memory

reload_strategy: hot
```

**prompts/system.md:**
```markdown
You are a helpful assistant that solves tasks step by step.
Use available tools to gather information before answering.
```

### Step 2: Implement BaseAgent

```python
from agent_evolve.harness.protocol.base_agent import BaseAgent
from agent_evolve.types import Task, Trajectory

class MyAgent(BaseAgent):
    def __init__(self, workspace_dir, model_id="us.anthropic.claude-opus-4-6-v1", **kwargs):
        super().__init__(workspace_dir)
        self.model_id = model_id

    def solve(self, task: Task) -> Trajectory:
        """The only method you need to implement."""

        # These are automatically loaded from your workspace files
        # and reloaded after each evolution cycle:
        prompt = self.system_prompt           # from prompts/system.md
        skills = self.skills                  # from skills/*/SKILL.md
        memories = self.memories              # from memory/*.jsonl

        # Build your prompt with evolved skills
        full_prompt = prompt
        for skill in skills:
            full_prompt += f"\n\n## Skill: {skill.name}\n{skill.description}"

        # Your solving logic here
        result = call_your_llm(
            system=full_prompt,
            user=task.input,
        )

        return Trajectory(
            task_id=task.id,
            output=result,
            steps=[],  # optional: trace of actions taken
        )
```

**Key things BaseAgent gives you for free:**
- `self.workspace` — `AgentWorkspace` for file I/O
- `self.system_prompt` — auto-loaded from `prompts/system.md`
- `self.skills` — auto-loaded from `skills/*/SKILL.md`
- `self.memories` — auto-loaded from `memory/*.jsonl`
- `reload_from_fs()` — called automatically between evolution cycles
- `remember(content, category)` — buffer memory entries

### Step 3: Evolve it

```python
import agent_evolve as ae

evolver = ae.Evolver(
    agent=MyAgent("./my_agent", model_id="us.anthropic.claude-opus-4-6-v1"),
    benchmark="mcp-atlas",  # or your own BenchmarkAdapter
)
results = evolver.run(cycles=10)

print(f"Final score: {results.final_score:.3f}")
print(f"Score history: {results.score_history}")
print(f"Converged: {results.converged}")
```

---

## Tutorial 4: Write a Custom Evolution Algorithm

All built-in algorithms implement one interface: `EvolutionEngine.step()`.

### Step 1: Implement EvolutionEngine

```python
from agent_evolve.harness.engine.base import EvolutionEngine
from agent_evolve.types import StepResult

class MyEvolver(EvolutionEngine):
    def step(self, workspace, observations, history, trial) -> StepResult:
        """
        Called once per evolution cycle.

        Args:
            workspace:    AgentWorkspace — read/write workspace files
            observations: list[Observation] — this batch's results
                          Each has: .task, .trajectory, .feedback
            history:      EvolutionHistory — query past cycles
                          history.get_scores() -> list[float]
            trial:        TrialRunner — validate changes on holdout tasks
                          trial.run(tasks) -> list[Feedback]

        Returns:
            StepResult(mutated=bool, summary=str)
        """

        # 1. Analyze what went wrong
        failures = [o for o in observations if not o.feedback.success]
        if not failures:
            return StepResult(mutated=False, summary="All passed, nothing to evolve")

        # 2. Extract patterns from failures
        error_patterns = []
        for obs in failures:
            error_patterns.append({
                "task": obs.task.id,
                "score": obs.feedback.score,
                "detail": obs.feedback.detail,
            })

        # 3. Generate a new skill using an LLM
        from agent_evolve.llm.anthropic import AnthropicProvider
        llm = AnthropicProvider(model="us.anthropic.claude-sonnet-4-20250514-v1:0")

        skill_content = llm.complete(
            f"Based on these failure patterns, write a SKILL.md that helps "
            f"an agent avoid these mistakes:\n{error_patterns}"
        )

        # 4. Write the skill to the workspace
        workspace.write_skill("failure-handler", skill_content)

        # 5. Optionally update the prompt
        current_prompt = workspace.read_prompt()
        workspace.write_prompt(
            current_prompt + "\n\nIMPORTANT: Check the failure-handler skill before answering."
        )

        # 6. Optionally add a memory entry
        workspace.add_memory({
            "cycle": len(history.get_scores()) + 1,
            "failures": len(failures),
            "action": "Added failure-handler skill",
        })

        return StepResult(
            mutated=True,
            summary=f"Added failure-handler skill based on {len(failures)} failures",
        )

    def on_cycle_end(self, accepted: bool, score: float) -> None:
        """Called after gating. Use for meta-learning."""
        if not accepted:
            print(f"Mutation rejected (score={score}), adjusting strategy...")
```

### Step 2: Use it

```python
evolver = ae.Evolver(
    agent="mcp",
    benchmark="mcp-atlas",
    engine=MyEvolver(config),
)
results = evolver.run(cycles=10)
```

### Built-in algorithms to study

| Algorithm | File | Strategy |
|-----------|------|----------|
| `AdaptiveEvolveEngine` | `algorithms/adaptive_evolve/` | Per-claim feedback + meta-learning |
| `GuidedSynthesisEngine` | `algorithms/guided_synth/` | Agent-proposed skills + LLM curation |
| `AdaptiveSkillEngine` | `algorithms/adaptive_skill/` | LLM with bash tool access |
| `SkillForgeEngine` | `algorithms/skillforge/` | LLM mutation + EGL gating |

---

## Benchmark-Specific Guides

### MCP-Atlas

**What**: Tool-calling tasks across 16+ MCP servers (search, databases, APIs).

**Setup**:
```bash
docker pull ghcr.io/scaleapi/mcp-atlas:latest
# Optional: create .env with API keys for more tasks
```

**Quick test** (2 tasks, no evolution):
```bash
python examples/mcp_examples/adaptive_evolve_baseline.py \
  --docker-image ghcr.io/scaleapi/mcp-atlas:latest \
  --limit 2 --batch-size 2 \
  --output-dir /tmp/mcp_test
```

**Full evolution**:
```bash
python examples/mcp_examples/adaptive_evolve_all.py \
  --docker-image ghcr.io/scaleapi/mcp-atlas:latest \
  --limit 50 --batch-size 10 \
  --seed-workspace seed_workspaces/mcp \
  --work-dir ./evolution_workdir/mcp \
  --output-dir results/mcp_evolved
```

### SWE-bench Verified

**What**: Real GitHub issues — the agent reads code, writes patches, runs tests.

**Setup**: Docker images are auto-pulled per task from `swebench/sweb.eval.x86_64.*`.

**Quick test** (2 tasks, no evolution):
```bash
python examples/swe_examples/solve_all.py \
  --limit 2 --workers 1 \
  --output-dir /tmp/swe_test
```

**Full evolution**:
```bash
python examples/swe_examples/evolve_sequential.py \
  --batch-size 5 --parallel 3 \
  --feedback minimal \
  --seed-workspace seed_workspaces/swe \
  --output-dir results/swe_evolved
```

### Terminal-Bench 2.0

**What**: CLI/shell tasks in Docker containers (89 challenges, multiple languages).

**Setup**: Challenge files auto-download from GitHub on first run (~200MB).

**Quick test** (2 tasks, no evolution):
```bash
BYPASS_TOOL_CONSENT=true python examples/tb_examples/batch_evolve_terminal.py \
  --solver react --no-evolve --no-skills \
  --workers 1 --limit 2 \
  --work-dir /tmp/tb2_test_ws \
  --log-dir /tmp/tb2_test \
  --output /tmp/tb2_test/results.jsonl \
  --errors /tmp/tb2_test/errors.jsonl
```

**Full evolution**:
```bash
bash examples/tb_examples/run_evolution.sh my_run_name
```

### SkillBench

**What**: Agentic skill discovery tasks.

**Setup**: Requires local task directories (not publicly available).

```bash
python examples/skillbench_examples/skillbench_evolve_in_situ_cycle.py \
  --batch-size 2 --max-cycles 3 \
  --tasks-dir-with-skills /path/to/skillsbench/tasks \
  --tasks-dir-without-skills /path/to/skillsbench/tasks-no-skills
```

---

## Configuration Reference

### EvolveConfig

```python
from agent_evolve.config import EvolveConfig

config = EvolveConfig(
    # How many tasks per batch
    batch_size=10,              # More = stabler signal, slower cycles

    # When to stop
    max_cycles=20,              # Max evolution cycles
    egl_window=3,               # Convergence: stop after 3 flat cycles
    egl_threshold=0.05,         # Min improvement to count as progress

    # What to evolve
    evolve_prompts=True,        # Mutate system prompt
    evolve_skills=True,         # Mutate skills
    evolve_memory=True,         # Mutate episodic memory
    evolve_tools=False,         # Mutate tool definitions (experimental)

    # Validation
    holdout_ratio=0.2,          # 20% of tasks reserved for gating

    # LLM for evolution
    evolver_model="us.anthropic.claude-opus-4-6-v1",
    evolver_max_tokens=16384,
)
```

### Model IDs (AWS Bedrock)

| Model | Bedrock ID |
|-------|-----------|
| Claude Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` |
| Claude Sonnet 4 | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| Claude Sonnet 4.5 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `AWS_DEFAULT_REGION` | AWS region for Bedrock | Yes |
| `AWS_ACCESS_KEY_ID` | AWS credentials | Yes (or use profile) |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | Yes (or use profile) |
| `BYPASS_TOOL_CONSENT` | Skip tool consent prompts (batch mode) | For Terminal-Bench |
| `TB2_CHALLENGES_DIR` | Override Terminal-Bench challenges path | No |

---

## Troubleshooting

### "No such file or directory: seed_workspaces/..."

The example scripts use relative paths. Always run from the repo root:
```bash
cd /path/to/a-evolve
python examples/mcp_examples/...
```

### "Cannot import 'SweVerifiedMiniBenchmark'"

The `__init__.py` for `swe_verified_mini` may be missing. Create it:
```bash
echo "from .benchmark import SweVerifiedMiniBenchmark" > \
  agent_evolve/benchmarks/swe_verified_mini/__init__.py
```

### Docker image pull failures

Pre-pull images before running:
```bash
docker pull ghcr.io/scaleapi/mcp-atlas:latest      # MCP-Atlas
# SWE-bench images are auto-pulled per task
```

### Evolution not converging

- Increase `batch_size` (more tasks = more stable signal)
- Increase `max_cycles` (give it more attempts)
- Widen `egl_window` (e.g., 5 instead of 3)
- Check if agent is actually using evolved skills (inspect `reload_from_fs()` output)

### "MCP-Atlas: all tasks filtered"

You need API keys for MCP servers. Without any `.env` file, only ~40% of tasks (keyless servers) are available. Add keys for Brave Search, Exa, etc. to unlock more.

### Resume interrupted runs

All example scripts support resume — they check for existing results and skip completed tasks:
```bash
# Just re-run the same command — completed tasks are skipped
python examples/mcp_examples/adaptive_evolve_all.py \
  --output-dir results/mcp_evolved  # same output dir
```

### High token usage

Terminal-Bench tasks can use 200K-1M tokens per task. To reduce costs during testing:
- Use `--limit 2` to test with fewer tasks
- Use Sonnet instead of Opus for the solver: `--solver-model us.anthropic.claude-sonnet-4-20250514-v1:0`
- Reduce `--max-tokens 8192` (default is 16384)

---

## What's Next

- Read `DESIGN.md` for architecture deep-dive
- Study `algorithms/adaptive_evolve/` for a production evolution engine
- Check `docs/` for benchmark-specific demo walkthroughs
- Paper: [arXiv 2602.00359](https://arxiv.org/abs/2602.00359)
