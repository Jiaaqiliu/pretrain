"""Tiny eval-smoke: instantiate NemoReasonerBenchmark + run build_eval_plan
against the host's seed adapter (if present) with limit=4.

Designed to be invoked inside a pod via `python run_eval_smoke.py <adapter_path>`.
Writes metrics.json into the workspace's evolution/eval directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/fsx/zzsamshi/a-evolve")
sys.path.insert(0, str(REPO))

from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend  # noqa: E402
from agent_evolve.benchmarks.nemo_reasoner import NemoReasonerBenchmark  # noqa: E402
from agent_evolve.model.types import CheckpointRef  # noqa: E402
from agent_evolve.model.workspace import TrainingWorkspace  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_eval_smoke.py <adapter_dir>", file=sys.stderr)
        return 2
    adapter_dir = Path(sys.argv[1])

    if not (adapter_dir / "adapter_config.json").is_file():
        print(
            f"adapter at {adapter_dir} has no adapter_config.json — cannot eval",
            file=sys.stderr,
        )
        return 2

    seed_ws_path = REPO / "seed_workspaces" / "nemotron_reasoner"
    # Force limit=4 in kaggle_eval.yaml via env override (primary metric
    # only looks at limit via the YAML; override through a patched copy).
    # Simplest: write a copy of kaggle_eval.yaml into a scratch workspace.
    import shutil
    import yaml

    smoke_root = Path(os.environ.get("AE_SMOKE_ROOT", "/fsx/zzsamshi/a-evolve-smoke"))
    eval_ws = smoke_root / "eval_ws"
    if eval_ws.exists():
        shutil.rmtree(eval_ws)
    shutil.copytree(seed_ws_path, eval_ws, symlinks=True)

    # Force limit=4 and tensor_parallel_size=1 so we fit in a single GPU pod.
    k = eval_ws / "eval" / "kaggle_eval.yaml"
    cfg = yaml.safe_load(k.read_text()) or {}
    cfg["limit"] = 4
    cfg["tensor_parallel_size"] = 1
    cfg["max_num_seqs"] = 16
    cfg["gpu_memory_utilization"] = 0.80
    k.write_text(yaml.safe_dump(cfg))

    workspace = TrainingWorkspace(root=str(eval_ws))
    benchmark = NemoReasonerBenchmark()
    backend = SingleNodeTinkerLiteBackend(mock=False)

    ckpt = CheckpointRef(
        name=os.environ.get("AE_SMOKE_CKPT_NAME", "smoke_adapter"),
        path=str(adapter_dir),
        kind="adapter",
        metadata={"source": "smoke"},
    )

    # Mirror the private state SingleNodeTinkerLiteBackend expects before
    # run_eval_plan(...).
    backend._current_workspace = workspace
    backend._current_benchmark = benchmark
    backend._current_split = "kaggle_dev_local"

    out = benchmark.evaluate(workspace, ckpt, backend, "kaggle_dev_local")
    print(f"[smoke-eval] wrote results to {out}")
    metrics = json.loads((Path(out) / "metrics.json").read_text())
    print(f"[smoke-eval] metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
