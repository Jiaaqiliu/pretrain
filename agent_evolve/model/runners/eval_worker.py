"""Pod-side entrypoint for k8s-dispatched eval jobs.

Invoked inside the trainer image as::

    python -m agent_evolve.model.runners.eval_worker --config <CFG_JSON>

``CFG_JSON`` lives on the FSx PVC so both the host scheduler and the pod
read the same bytes. Payload shape::

    {
      "plan": <EvalPlan-as-dict>,
      "workspace_root": "/fsx/...",
      "benchmark_name": "nemo_reasoner",
      "split": "kaggle_dev_local",
      "out_result_path": "/fsx/.../.eval_result.json"
    }

The worker reconstructs the benchmark by name (we only support
``nemo_reasoner`` today — extend the registry below as new benchmarks
come online), reconstructs an ``EvalPlan`` dataclass, and calls the
same ``run_eval_plan`` function the host path uses. On success it
writes ``{"status": "ok", "output_dir": ...}`` at ``out_result_path``
and exits 0. On exception it writes ``{"status": "failed",
"error": ...}`` and exits non-zero so the scheduler flags the job
as failed (not "succeeded with a bad result").
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def _import_benchmark(name: str):
    if name == "nemo_reasoner":
        from agent_evolve.benchmarks.nemo_reasoner import NemoReasonerBenchmark
        return NemoReasonerBenchmark()
    raise ValueError(
        f"eval_worker: unknown benchmark {name!r}. "
        "Add it to _import_benchmark in eval_worker.py."
    )


def _rebuild_plan(plan_dict: dict):
    from agent_evolve.model.types import CheckpointRef, EvalPlan
    ckpt_dict = plan_dict.get("checkpoint") or {}
    checkpoint = CheckpointRef(
        name=ckpt_dict["name"],
        path=ckpt_dict["path"],
        kind=ckpt_dict.get("kind", "adapter"),
        metadata=ckpt_dict.get("metadata") or {},
    )
    return EvalPlan(
        benchmark_name=plan_dict["benchmark_name"],
        split=plan_dict["split"],
        checkpoint=checkpoint,
        config_path=plan_dict["config_path"],
        output_dir=plan_dict["output_dir"],
        generation_config=plan_dict.get("generation_config") or {},
        metadata=plan_dict.get("metadata") or {},
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to JSON config.")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())

    plan_dict = cfg["plan"]
    workspace_root = cfg["workspace_root"]
    benchmark_name = cfg.get("benchmark_name") or plan_dict.get("benchmark_name")
    split = cfg.get("split") or plan_dict.get("split")
    out_result_path = Path(cfg["out_result_path"])

    # Lazy import so argparse failures don't take the torch stack with them.
    from agent_evolve.model.runners.stages.eval import run_eval_plan

    plan = _rebuild_plan(plan_dict)
    benchmark = _import_benchmark(benchmark_name)

    # The stage worker only touches ``workspace.root``; a SimpleNamespace
    # is enough. Reuse the same FSx path the host sees.
    workspace = SimpleNamespace(root=workspace_root)

    logger.info(
        "[eval_worker] start cfg=%s benchmark=%s split=%s ckpt=%s out=%s",
        cfg_path, benchmark_name, split, plan.checkpoint.path, plan.output_dir,
    )

    try:
        out_dir = run_eval_plan(
            plan, smoke=False, benchmark=benchmark,
            workspace=workspace, split=split,
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("[eval_worker] run_eval_plan raised: %s\n%s", exc, tb)
        out_result_path.parent.mkdir(parents=True, exist_ok=True)
        out_result_path.write_text(json.dumps({
            "status": "failed",
            "error": repr(exc),
            "traceback": tb,
        }, indent=2))
        return 1

    out_result_path.parent.mkdir(parents=True, exist_ok=True)
    out_result_path.write_text(json.dumps({
        "status": "ok",
        "output_dir": str(out_dir),
    }, indent=2))
    logger.info("[eval_worker] done output_dir=%s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
