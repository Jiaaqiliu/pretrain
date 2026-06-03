"""CLI entry point for HoloPretrain.

Usage:
    # Run the orchestrator with 3 trials
    python -m autopretrain.cli run \
        --trials llama3,reasoning_heavy,uniform \
        --model olmo2_1B \
        --steps 5000

    # Check status of running experiments
    python -m autopretrain.cli status

    # View recent events
    python -m autopretrain.cli events --tail 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from autopretrain.core.types import Budget, DataMixSpec, TrialConfig
from autopretrain.compute.kubernetes import K8sConfig, KubernetesBackend
from autopretrain.engine.olmo_adapter import OLMoAdapter
from autopretrain.orchestrator.orchestrator import Orchestrator, OrchestratorConfig
from autopretrain.store.event_log import EventLog

logger = logging.getLogger(__name__)

# Predefined data mixes
PREDEFINED_MIXES = {
    "llama3": {"web": 0.52, "code": 0.18, "math": 0.26, "academic": 0.04},
    "reasoning_heavy": {"web": 0.25, "code": 0.30, "math": 0.30, "academic": 0.15},
    "uniform": {"web": 0.25, "code": 0.25, "math": 0.25, "academic": 0.25},
    "code_heavy": {"web": 0.20, "code": 0.40, "math": 0.25, "academic": 0.15},
    "academic_heavy": {"web": 0.30, "code": 0.15, "math": 0.20, "academic": 0.35},
}

DOMAIN_PATHS = {
    "web": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/dclm_web/*.bin",
    "code": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/code/*.bin",
    "math": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/math/*.bin",
    "academic": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/fineweb_edu/*.bin",
}

CHECKPOINT_BASE = "/fsx/dev/jiaqi/checkpoints/autopretrain"


def build_trial_configs(args) -> list[TrialConfig]:
    """Build trial configurations from CLI arguments."""
    trial_names = [t.strip() for t in args.trials.split(",")]
    configs = []

    for name in trial_names:
        mix = PREDEFINED_MIXES.get(name)
        if mix is None:
            logger.error("Unknown mix: %s. Available: %s", name, list(PREDEFINED_MIXES.keys()))
            sys.exit(1)

        data_mix = DataMixSpec(
            sources=mix,
            paths=DOMAIN_PATHS,
            max_repetition_ratio=4.0,
        )

        config = TrialConfig(
            name=name,
            data_mix=data_mix,
            model_factory=args.model,
            max_steps=args.steps,
            global_batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.warmup,
            sequence_length=args.seq_len,
            ephemeral_save_interval=args.eph_ckpt_interval,
            save_interval=args.steps,
            save_folder=f"{CHECKPOINT_BASE}/{name}",
            use_skip_step_optimizer=not args.no_skip_step,
            deadline_seconds=args.deadline,
        )
        configs.append(config)

    return configs


async def cmd_run(args):
    """Run the orchestrator with specified trials."""
    trials = build_trial_configs(args)

    logger.info("Launching %d trials: %s", len(trials), [t.name for t in trials])

    # Build components
    k8s = KubernetesBackend(K8sConfig(
        namespace=args.namespace,
        context=args.context,
    ))
    engine = OLMoAdapter(
        heartbeat_dir=args.heartbeat_dir,
    )

    orch_config = OrchestratorConfig(
        poll_interval=args.poll_interval,
        max_concurrent_trials=args.max_concurrent,
        heartbeat_dir=args.heartbeat_dir,
        budget=Budget(
            max_gpu_hours=args.max_gpu_hours,
            max_trials=len(trials) * 3,  # Allow retries
        ),
    )

    orchestrator = Orchestrator(orch_config, k8s, engine)
    results = await orchestrator.run(trials)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for r in results:
        print(f"  {r.trial_id} ({r.config.name}): loss={r.final_loss or 'N/A'}, step={r.final_step}, status={r.status}")
    print("=" * 60)

    return results


def cmd_status(args):
    """Show status of recent events."""
    event_log = EventLog(Path(args.event_log))
    events = event_log.tail(n=args.tail)

    if not events:
        print("No events found.")
        return

    print(f"Last {len(events)} events:")
    print("-" * 80)
    for e in events:
        import datetime
        ts = datetime.datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
        print(f"  [{ts}] {e.trial_id}: {e.from_state} → {e.to_state} ({e.event_type})")


def cmd_events(args):
    """View detailed events."""
    event_log = EventLog(Path(args.event_log))
    events = event_log.tail(n=args.tail)

    for e in events:
        print(json.dumps({
            "timestamp": e.timestamp,
            "trial": e.trial_id,
            "type": e.event_type,
            "from": e.from_state,
            "to": e.to_state,
            "details": e.details,
        }, indent=2))
        print()


def main():
    parser = argparse.ArgumentParser(description="HoloPretrain: Autonomous Training Framework")
    parser.add_argument("--verbose", "-v", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run training trials")
    run_parser.add_argument("--trials", type=str, required=True,
                           help="Comma-separated trial names (e.g., llama3,reasoning_heavy,uniform)")
    run_parser.add_argument("--model", type=str, default="olmo2_1B",
                           help="Model factory (olmo2_190M, olmo2_1B, olmo2_3B, etc.)")
    run_parser.add_argument("--steps", type=int, default=5000)
    run_parser.add_argument("--batch-size", type=int, default=128)
    run_parser.add_argument("--lr", type=float, default=3e-4)
    run_parser.add_argument("--warmup", type=int, default=500)
    run_parser.add_argument("--seq-len", type=int, default=4096)
    run_parser.add_argument("--eph-ckpt-interval", type=int, default=500)
    run_parser.add_argument("--deadline", type=int, default=14400)
    run_parser.add_argument("--max-concurrent", type=int, default=3)
    run_parser.add_argument("--max-gpu-hours", type=float, default=100.0)
    run_parser.add_argument("--poll-interval", type=float, default=30.0)
    run_parser.add_argument("--no-skip-step", action="store_true")
    run_parser.add_argument("--namespace", default="default")
    run_parser.add_argument("--context", default="arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm")
    run_parser.add_argument("--heartbeat-dir", default="/fsx/dev/jiaqi/experiments/autopretrain/heartbeat")
    run_parser.add_argument("--event-log", default="/fsx/dev/jiaqi/experiments/autopretrain/agent_logs/events.jsonl")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show experiment status")
    status_parser.add_argument("--tail", type=int, default=20)
    status_parser.add_argument("--event-log", default="/fsx/dev/jiaqi/experiments/autopretrain/agent_logs/events.jsonl")

    # Events command
    events_parser = subparsers.add_parser("events", help="View detailed events")
    events_parser.add_argument("--tail", type=int, default=10)
    events_parser.add_argument("--event-log", default="/fsx/dev/jiaqi/experiments/autopretrain/agent_logs/events.jsonl")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "events":
        cmd_events(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
