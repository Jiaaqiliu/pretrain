#!/usr/bin/env python3
"""Evolve an ARC-AGI-3 agent through interactive game play.

Uses a-evolve to iteratively improve the agent's prompts, skills, and
memory for playing ARC-AGI-3 games with maximum efficiency (RHAE).

Prerequisites:
    pip install arc-agi

Usage:
    # Quick start with defaults
    python evolve_arc.py

    # Custom configuration
    python evolve_arc.py --cycles 20 --batch-size 3 --engine adaptive-skill

    # Offline mode (local games only, no API key needed)
    python evolve_arc.py --operation-mode offline

    # With API key for full game catalog
    python evolve_arc.py --api-key YOUR_KEY --operation-mode competition
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import agent_evolve as ae
from agent_evolve.config import EvolveConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("evolve_arc")


def main():
    parser = argparse.ArgumentParser(description="Evolve ARC-AGI-3 agent")
    parser.add_argument(
        "--cycles", type=int, default=10,
        help="Number of evolution cycles (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=3,
        help="Games per batch (default: 3, ARC games are long-running)",
    )
    parser.add_argument(
        "--engine", default="default",
        choices=["default", "adaptive-evolve", "adaptive-skill", "guided-synth"],
        help="Evolution engine (default: AEvolveEngine)",
    )
    parser.add_argument(
        "--model", default="us.anthropic.claude-opus-4-6-v1",
        help="Model ID for the agent",
    )
    parser.add_argument(
        "--operation-mode", default="normal",
        choices=["normal", "offline", "online", "competition"],
        help="ARC-AGI-3 operation mode (default: normal)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="ARC API key (or set ARC_API_KEY env var)",
    )
    parser.add_argument(
        "--max-actions", type=int, default=5000,
        help="Max actions per game (default: 5000)",
    )
    parser.add_argument(
        "--game-filter", default=None,
        help="Only include games whose ID contains this substring",
    )
    parser.add_argument(
        "--output-dir", default="./evolution_workdir",
        help="Working directory for evolution state",
    )
    args = parser.parse_args()

    config = EvolveConfig(
        batch_size=args.batch_size,
        max_cycles=args.cycles,
        evolve_prompts=True,
        evolve_skills=True,
        evolve_memory=True,
    )

    # Select evolution engine
    engine = None
    if args.engine == "adaptive-evolve":
        from agent_evolve.harness.algorithms.adaptive_evolve.engine import AdaptiveEvolveEngine
        engine = AdaptiveEvolveEngine(config)
    elif args.engine == "adaptive-skill":
        from agent_evolve.harness.algorithms.adaptive_skill.engine import AdaptiveSkillEngine
        engine = AdaptiveSkillEngine(config)
    elif args.engine == "guided-synth":
        from agent_evolve.harness.algorithms.guided_synth.engine import GuidedSynthesisEngine
        engine = GuidedSynthesisEngine(config)

    # Create benchmark with ARC-specific settings
    from agent_evolve.benchmarks.arc_agi3 import ArcAgi3Benchmark

    benchmark = ArcAgi3Benchmark(
        api_key=args.api_key,
        operation_mode=args.operation_mode,
        game_filter=args.game_filter,
        max_actions_per_game=args.max_actions,
    )

    logger.info("Starting ARC-AGI-3 evolution")
    logger.info("  Cycles: %d", args.cycles)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Engine: %s", args.engine)
    logger.info("  Model: %s", args.model)
    logger.info("  Operation mode: %s", args.operation_mode)
    logger.info("  Max actions/game: %d", args.max_actions)

    evolver = ae.Evolver(
        agent="arc",
        benchmark=benchmark,
        config=config,
        engine=engine,
        work_dir=args.output_dir,
    )

    results = evolver.run(cycles=args.cycles)

    logger.info("Evolution complete!")
    logger.info("  Cycles completed: %d", results.cycles_completed)
    logger.info("  Final score: %.3f", results.final_score)
    logger.info("  Score history: %s", results.score_history)
    logger.info("  Converged: %s", results.converged)

    return results


if __name__ == "__main__":
    main()
