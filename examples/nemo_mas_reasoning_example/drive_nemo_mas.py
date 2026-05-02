#!/usr/bin/env python3
"""Driver for the nemo_mas algorithm on the Nemotron Reasoning workspace.

Three modes (--mode):

  * ``dry-run`` (default): no Bedrock, no GPU. Monkey-patches BedrockAgent
    with a scripted stub so the whole loop runs in seconds. Useful for
    validating the workspace + memory + tool wiring.

  * ``demo``: enables the demo backend handlers (plausible mock outputs
    for run_eval / launch_training / call_teacher_model). Still requires
    Bedrock for the orchestrator + workers — set AWS creds first.

  * ``real``: wires in SingleNodeTinkerLiteBackend(mock=False) +
    NemoReasonerBenchmark for actual training. Expects a configured
    backend and GPUs.

Usage::

    .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
        --cycles 3 --mode dry-run

    .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
        --cycles 5 --mode demo --workspace seed_workspaces/nemo_mas_reasoner

    .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
        --cycles 10 --mode real \
        --workspace seed_workspaces/nemo_mas_reasoner \
        --work-dir runs/nemo-mas-10
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

# Make the repo importable when run as a script.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def install_dry_run_bedrock_stub(scripted_responses: list[str]) -> None:
    """Inject a stub BedrockAgent that pops scripted responses on call().

    Calls past the end of the script return a generic message. Use
    --print-prompts to dump what the orchestrator sent.
    """
    queue = list(scripted_responses)
    log: list[dict] = []

    class StubBedrockAgent:
        def __init__(self, *, model_id, system_prompt, tools, tool_handlers,
                     agent_id=0, max_tokens=16384, thinking_effort="",
                     **_kw):
            self.model_id = model_id
            self.system_prompt = system_prompt
            self.tools = tools
            self.tool_handlers = tool_handlers
            self.agent_id = agent_id

        def call(self, task: str) -> str:
            log.append({
                "agent_id": self.agent_id,
                "model_id": self.model_id,
                "task_excerpt": task[:200],
                "n_tools": len(self.tools),
            })
            return queue.pop(0) if queue else (
                "(stub orchestrator: nothing more to do)"
            )

    fake = types.ModuleType("agent_evolve.harness.agents.arc.bedrock_agent")
    fake.BedrockAgent = StubBedrockAgent
    sys.modules["agent_evolve.harness.agents.arc.bedrock_agent"] = fake
    # Stash the log on the module so the driver can dump it later.
    fake._log = log


def build_algorithm(*, mode: str, workspace_root: Path):
    """Construct NemoMASAlgorithm with the right backend_registry for the mode."""
    from agent_evolve.model.algorithms.nemo_mas import NemoMASAlgorithm
    from agent_evolve.model.algorithms.nemo_mas.backends import (
        BackendBridge, demo_compute_handlers, local_handlers,
    )

    registry: dict = {**local_handlers(workspace_root)}

    if mode == "dry-run":
        # No real compute; demo handlers + local file ops are enough to
        # exercise the loop.
        registry.update(demo_compute_handlers())
    elif mode == "demo":
        registry.update(demo_compute_handlers())
    elif mode == "real":
        from agent_evolve.benchmarks.nemo_reasoner import NemoReasonerBenchmark
        from agent_evolve.backends.tinkerlite.single_node.backend import (
            SingleNodeTinkerLiteBackend,
        )
        backend = SingleNodeTinkerLiteBackend(mock=False)
        benchmark = NemoReasonerBenchmark()
        bridge = BackendBridge(workspace_root=workspace_root,
                               benchmark=benchmark, backend=backend)
        registry.update(bridge.as_registry())
    else:
        raise ValueError(f"unknown mode: {mode}")

    return NemoMASAlgorithm(backend_registry=registry)


def run_via_evolver(*, workspace: Path, work_dir: Path, cycles: int,
                    algo, smoke: bool, trial_budget_seconds: float):
    """Use TrainingEvolver if available; fall back to direct run_cycle loop."""
    try:
        from agent_evolve.api import TrainingEvolver  # type: ignore
        from agent_evolve.model.types import TrainingEvolveConfig  # type: ignore
    except Exception:                              # noqa: BLE001
        TrainingEvolver = None                     # type: ignore
        TrainingEvolveConfig = None                # type: ignore

    if TrainingEvolver is not None and TrainingEvolveConfig is not None:
        evolver = TrainingEvolver(
            workspace=workspace,
            benchmark="nemo_reasoner",
            algorithm=algo,
            backend="h200_single_node",
            config=TrainingEvolveConfig(
                smoke=smoke,
                max_cycles=cycles,
                trial_budget_seconds=trial_budget_seconds,
            ),
            work_dir=work_dir,
        )
        result = evolver.run(cycles=cycles)
        return {
            "kind": "evolver",
            "cycles_completed": getattr(result, "cycles_completed", None),
            "best_metric": getattr(result, "best_metric", None),
            "incumbent_node_id": getattr(result, "incumbent_node_id", None),
        }

    # Fallback: directly drive run_cycle with a synthetic LoopContext.
    print("[driver] TrainingEvolver unavailable; running direct loop.",
          file=sys.stderr)
    work_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for c in range(1, cycles + 1):
        ctx = types.SimpleNamespace(
            cycle=c, workspace=workspace, benchmark=None, backend=None,
            config=None, work_dir=work_dir, trial=None, observer=None,
            budget=types.SimpleNamespace(seconds=trial_budget_seconds,
                                         steps=None, tokens=None),
        )
        report = algo.run_cycle(ctx)
        reports.append({
            "cycle": report.cycle,
            "trial_node_ids": report.trial_node_ids,
            "incumbent_changed": report.incumbent_changed,
            "best_metric": report.best_metric,
        })
    return {
        "kind": "direct",
        "cycles_completed": cycles,
        "best_metric": reports[-1]["best_metric"] if reports else None,
        "incumbent_node_id": None,
        "per_cycle": reports,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--mode", choices=("dry-run", "demo", "real"),
                    default="dry-run")
    ap.add_argument("--workspace", default="seed_workspaces/nemo_mas_reasoner",
                    help="Path to the nemo_mas workspace (default: seed).")
    ap.add_argument("--work-dir", default=None,
                    help="Output directory for cycle artifacts (default: "
                         "runs/nemo-mas-<mode>).")
    ap.add_argument("--trial-budget-seconds", type=float, default=300.0)
    ap.add_argument("--smoke", action="store_true",
                    help="Pass smoke=True to TrainingEvolveConfig "
                         "(short-circuits real training).")
    ap.add_argument("--print-prompts", action="store_true",
                    help="Dump the prompts seen by the stub BedrockAgent "
                         "(dry-run mode only).")
    ap.add_argument("--script", action="append", default=None,
                    help="Scripted orchestrator response (dry-run mode). "
                         "Repeat to script multiple cycles.")
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    if not workspace.exists():
        print(f"[driver] workspace not found: {workspace}", file=sys.stderr)
        return 2

    work_dir = Path(args.work_dir or f"runs/nemo-mas-{args.mode}").resolve()

    # Dry-run: install the stub BEFORE importing the algorithm package so
    # the spawner's lazy import resolves to the stub.
    if args.mode == "dry-run":
        scripted = args.script or [
            f"Cycle stub response — no spawns this run (cycle {i + 1})."
            for i in range(args.cycles)
        ]
        install_dry_run_bedrock_stub(scripted)

    algo = build_algorithm(mode=args.mode, workspace_root=workspace)

    print(f"[driver] mode={args.mode} cycles={args.cycles} "
          f"workspace={workspace}")
    print(f"[driver] backend tools wired: {len(algo.backend_registry or {})}")

    summary = run_via_evolver(
        workspace=workspace, work_dir=work_dir, cycles=args.cycles,
        algo=algo,
        smoke=args.smoke or args.mode != "real",
        trial_budget_seconds=args.trial_budget_seconds,
    )

    print("\n[driver] === Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    if args.mode == "dry-run" and args.print_prompts:
        fake_module = sys.modules.get(
            "agent_evolve.harness.agents.arc.bedrock_agent"
        )
        log = getattr(fake_module, "_log", []) if fake_module else []
        print("\n[driver] === BedrockAgent stub call log ===")
        print(json.dumps(log, indent=2))

    # Show the records that landed in memory this run.
    records_path = workspace / "memory" / "records.jsonl"
    if records_path.exists():
        n_records = sum(
            1 for _ in records_path.open("r", encoding="utf-8")
        )
        print(f"\n[driver] memory now has {n_records} records "
              f"at {records_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
