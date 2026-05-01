"""CLI entrypoint — ``python -m agent_evolve.model.run``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .api import TrainingEvolver
from .types import TrainingEvolveConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the TrainingEvolver loop.")
    p.add_argument("--workspace", required=True, help="Path to training workspace.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--algorithm", default="mcgs")
    p.add_argument("--backend", default="h200_single_node")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--work-dir", default="./training_evolution_workdir")
    p.add_argument("--smoke", action="store_true", help="Bypass GPU-bound code paths.")
    p.add_argument("--log-level", default="INFO")
    return p


def _serialize_result(result) -> dict:
    data = asdict(result)
    return data


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config = TrainingEvolveConfig(smoke=args.smoke, max_cycles=args.cycles)
    evolver = TrainingEvolver(
        workspace=Path(args.workspace),
        benchmark=args.benchmark,
        algorithm=args.algorithm,
        backend=args.backend,
        config=config,
        work_dir=Path(args.work_dir),
    )
    result = evolver.run(cycles=args.cycles)
    print(json.dumps(_serialize_result(result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
