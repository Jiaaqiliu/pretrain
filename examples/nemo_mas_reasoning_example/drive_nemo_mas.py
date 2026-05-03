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
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path

# Make the repo importable when run as a script.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def configure_logging() -> None:
    """Route nemo_mas + BedrockAgent logs to stderr at NEMO_MAS_LOG level.

    ``bedrock_agent`` uses the root logger (``logging.getLogger()``);
    ``orchestrator`` and ``spawner`` use ``__name__`` loggers. Without
    ``basicConfig`` both are silent above WARNING — so retries, tool
    errors, and agent-construction failures never surface.
    """
    level = os.environ.get("NEMO_MAS_LOG", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def install_trace_wrapper(trace_dir: Path) -> None:
    """Wrap the real BedrockAgent so every turn is written to JSONL on disk.

    Writes ``<trace_dir>/cycle_<NNNN>/agent_<id>.jsonl`` — one line per
    turn (user, assistant, tool_results) plus a final ``event=done``
    line with usage + total turns. Does not touch ``bedrock_agent.py``;
    replaces the class in ``sys.modules`` the same way
    :func:`install_dry_run_bedrock_stub` does.

    Uses a module-level cycle counter that bumps on each top-level
    agent construction — the spawner constructs one ``BedrockAgent``
    per worker, and the orchestrator constructs one per cycle. We key
    files by a monotonic ``cycle_<NNNN>`` stamp so all agents spawned
    inside a cycle land together.
    """
    import importlib

    real_mod = importlib.import_module("agent_evolve.harness.agents.arc.bedrock_agent")
    RealBedrockAgent = real_mod.BedrockAgent

    trace_dir.mkdir(parents=True, exist_ok=True)
    # Orchestrator (agent_id=0) starts a new cycle. Worker agents
    # (agent_id>=1) share the current cycle_dir.
    state = {"cycle": 0, "cycle_dir": trace_dir}

    class TracingBedrockAgent(RealBedrockAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.agent_id == 0:
                state["cycle"] += 1
                state["cycle_dir"] = trace_dir / f"cycle_{state['cycle']:04d}"
                state["cycle_dir"].mkdir(parents=True, exist_ok=True)
            self._trace_path = state["cycle_dir"] / f"agent_{self.agent_id}.jsonl"
            self._trace_last_msg_count = 0
            self._trace_turn = 0
            # Header line: system prompt + tool names.
            tool_names = []
            for t in (self.tools or []):
                spec = t.get("toolSpec") if isinstance(t, dict) else None
                if spec:
                    tool_names.append(spec.get("name", "?"))
            self._trace_write({
                "event": "start",
                "agent_id": self.agent_id,
                "model_id": self.model_id,
                "system_excerpt": (self.system[0].get("text", "")[:500]
                                   if self.system else ""),
                "tool_names": tool_names,
            })

        def _trace_write(self, row: dict) -> None:
            row = {"ts": time.time(), **row}
            try:
                with self._trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            except OSError:
                pass  # tracing must never break the run

        def _run_converse_loop(self) -> str:
            # Log the new user message that ``call`` just appended.
            if len(self.messages) > self._trace_last_msg_count:
                for i in range(self._trace_last_msg_count, len(self.messages)):
                    m = self.messages[i]
                    self._trace_write({
                        "event": "message",
                        "turn": self._trace_turn,
                        "role": m.get("role"),
                        "content": m.get("content"),
                    })
                self._trace_last_msg_count = len(self.messages)

            # Drive one converse turn at a time by intercepting after each
            # client.converse round. Cheapest hook: monkey-patch just this
            # instance's ``_converse_with_retry`` to write the assistant
            # message + usage as it lands.
            real_converse = self._converse_with_retry

            def traced_converse(tool_config):
                self._trace_turn += 1
                result = real_converse(tool_config)
                try:
                    self._trace_write({
                        "event": "turn",
                        "turn": self._trace_turn,
                        "stop_reason": result.get("stopReason"),
                        "assistant": result.get("output", {}).get("message"),
                        "usage": result.get("usage"),
                    })
                except Exception:  # noqa: BLE001 — tracing is best-effort
                    pass
                return result

            self._converse_with_retry = traced_converse  # type: ignore[method-assign]

            try:
                result_text = super()._run_converse_loop()
            finally:
                self._trace_write({
                    "event": "done",
                    "total_turns": self._trace_turn,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "cache_read_tokens": self.total_cache_read_tokens,
                    "cache_write_tokens": self.total_cache_write_tokens,
                })
            return result_text

    real_mod.BedrockAgent = TracingBedrockAgent


class Heartbeat:
    """Every 30s, print one line summarizing cycle progress.

    Wall-clock activity pointer for long orchestrator calls. Reads the
    workspace's ``memory/records.jsonl`` line count and the algorithm's
    ``_last_cycle_records`` attribute between prints. Safe to stop
    mid-cycle — the daemon thread exits when :meth:`stop` is called.
    """

    def __init__(self, *, workspace: Path, algo, cycle: int,
                 interval_seconds: float = 30.0):
        self._workspace = workspace
        self._algo = algo
        self._cycle = cycle
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0

    def _records_path(self) -> Path:
        return self._workspace / "memory" / "records.jsonl"

    def _records_snapshot(self) -> tuple[int, str]:
        """(total line count, id of last record) — O(file size), OK for small jsonl."""
        p = self._records_path()
        if not p.exists():
            return 0, "-"
        try:
            count = 0
            last_id = "-"
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    count += 1
                    try:
                        last_id = json.loads(line).get("id", last_id)
                    except json.JSONDecodeError:
                        pass
            return count, last_id
        except OSError:
            return 0, "-"

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            elapsed = time.time() - self._t0
            mm, ss = divmod(int(elapsed), 60)
            count, last = self._records_snapshot()
            print(
                f"[heartbeat] cycle={self._cycle} elapsed={mm:02d}:{ss:02d} "
                f"records={count} last_record={last}",
                file=sys.stderr, flush=True,
            )

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


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


def build_algorithm(*, mode: str, workspace_root: Path, backend_kind: str = "local"):
    """Construct NemoMASAlgorithm with the right backend_registry for the mode.

    ``backend_kind`` is only consulted when ``mode == "real"``:
      * ``"local"``  → SingleNodeTinkerLiteBackend on the local GPUs.
      * ``"k8s"``    → K8sTinkerLiteBackend capped at 2 concurrent Jobs,
                        ``local_enabled=False`` so missing kubeconfig fails
                        fast instead of silently falling back to local.

    For dry-run / demo modes the arg is ignored — no real backend is used.
    """
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

        if backend_kind == "local":
            from agent_evolve.backends.tinkerlite.single_node.backend import (
                SingleNodeTinkerLiteBackend,
            )
            backend = SingleNodeTinkerLiteBackend(mock=False)
        elif backend_kind == "k8s":
            from agent_evolve.backends.tinkerlite.elastic.backend import (
                K8sTinkerLiteBackend,
            )
            backend = K8sTinkerLiteBackend(
                namespace=os.environ.get("AE_K8S_NAMESPACE", "a-evolve"),
                image=os.environ.get("AE_K8S_IMAGE", "a-evolve/trainer:latest"),
                pvc_name=os.environ.get("AE_K8S_PVC", "fsx-zzsamshi"),
                node_selector=(
                    {"nvidia.com/gpu.product": "H200"}
                    if os.environ.get("AE_K8S_NODE_LABEL", "1") == "1"
                    else None
                ),
                # Hard constraints from the plan: concurrent Jobs capped via
                # AE_K8S_QUEUE_BUDGET (default 2), no local fallback. Missing
                # kubeconfig / namespace raises at construction rather than
                # silently running on local GPUs.
                local_enabled=False,
                k8s_queue_budget=int(os.environ.get("AE_K8S_QUEUE_BUDGET", "2")),
                queue_timeout_secs=float(
                    os.environ.get("AE_K8S_QUEUE_TIMEOUT", "900")
                ),
            )
        else:
            raise ValueError(
                f"unknown backend_kind {backend_kind!r}; expected 'local' or 'k8s'"
            )

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
    # Wrap the seed as a TrainingWorkspace so run_cycle can call
    # ``workspace.fork(...)`` to isolate each cycle under
    # ``work_dir/cycles/<id>/workspace`` — keeps the seed read-only and
    # makes cycle outputs inspectable side-by-side.
    print("[driver] TrainingEvolver unavailable; running direct loop.",
          file=sys.stderr)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        from agent_evolve.model.workspace import TrainingWorkspace
        ws_obj = TrainingWorkspace(workspace)
    except Exception as exc:                     # noqa: BLE001
        print(f"[driver] could not wrap workspace ({exc!r}); "
              "falling back to raw-path mode (no per-cycle fork).",
              file=sys.stderr)
        ws_obj = workspace

    reports = []
    # Cycle counter bumps ONLY on cycles that make progress; `null` cycles
    # are retried up to `null_cycle_retry_limit` times. This is the
    # cycle-contract (proposal d) driver-side implementation.
    null_cycle_retry_limit = 3
    consecutive_nulls = 0
    produced = 0
    attempt = 0
    while produced < cycles:
        attempt += 1
        ctx = types.SimpleNamespace(
            cycle=produced + 1, workspace=ws_obj, benchmark=None, backend=None,
            config=None, work_dir=work_dir, trial=None, observer=None,
            budget=types.SimpleNamespace(seconds=trial_budget_seconds,
                                         steps=None, tokens=None),
        )
        with Heartbeat(workspace=workspace, algo=algo, cycle=produced + 1):
            report = algo.run_cycle(ctx)

        outcome = getattr(report, "cycle_outcome", "trained")
        reports.append({
            "cycle": report.cycle,
            "attempt": attempt,
            "outcome": outcome,
            "wall_seconds": getattr(report, "wall_seconds", None),
            "orchestrator_turns": getattr(report, "orchestrator_turns", None),
            "record_counts": getattr(report, "record_counts", {}),
            "trial_node_ids": report.trial_node_ids,
            "incumbent_changed": report.incumbent_changed,
            "best_metric": report.best_metric,
        })
        print(
            f"[driver] cycle attempt {attempt} → outcome={outcome} "
            f"records={len(report.trial_node_ids)} "
            f"incumbent_changed={report.incumbent_changed}",
            file=sys.stderr, flush=True,
        )

        if outcome == "null":
            consecutive_nulls += 1
            if consecutive_nulls >= null_cycle_retry_limit:
                print(
                    f"[driver] {consecutive_nulls} consecutive null cycles — "
                    f"aborting to avoid burning Bedrock on no progress.",
                    file=sys.stderr,
                )
                break
            print(
                f"[driver] null cycle; retrying ({consecutive_nulls}/"
                f"{null_cycle_retry_limit})",
                file=sys.stderr,
            )
            continue

        consecutive_nulls = 0
        produced += 1

    return {
        "kind": "direct",
        "cycles_completed": produced,
        "total_attempts": attempt,
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
    ap.add_argument("--trace-dir", default=None,
                    help="Write per-turn JSONL traces of every BedrockAgent "
                         "to this directory (demo / real modes). One file "
                         "per agent per cycle.")
    ap.add_argument("--backend", choices=("local", "k8s"), default="local",
                    help="Compute target for real mode. `local` (default) = "
                         "single-node torchrun on local GPUs. `k8s` = elastic "
                         "k8s submission capped at 2 concurrent Jobs, with no "
                         "local fallback (bad kubeconfig fails fast).")
    args = ap.parse_args()

    configure_logging()

    if args.backend == "k8s" and args.mode != "real":
        print(f"[driver] --backend k8s requires --mode real (got {args.mode!r})",
              file=sys.stderr)
        return 2

    workspace = Path(args.workspace).resolve()
    if not workspace.exists():
        print(f"[driver] workspace not found: {workspace}", file=sys.stderr)
        return 2

    default_work_suffix = (
        f"{args.mode}-{args.backend}" if args.mode == "real" else args.mode
    )
    work_dir = Path(args.work_dir or f"runs/nemo-mas-{default_work_suffix}").resolve()

    # Dry-run: install the stub BEFORE importing the algorithm package so
    # the spawner's lazy import resolves to the stub.
    if args.mode == "dry-run":
        scripted = args.script or [
            f"Cycle stub response — no spawns this run (cycle {i + 1})."
            for i in range(args.cycles)
        ]
        install_dry_run_bedrock_stub(scripted)
    elif args.trace_dir:
        # Install the tracing wrapper for demo / real modes. Must happen
        # before the spawner / orchestrator imports the class (though
        # BedrockAgent is lazy-imported inside call sites, so we're safe).
        install_trace_wrapper(Path(args.trace_dir).resolve())

    algo = build_algorithm(
        mode=args.mode, workspace_root=workspace, backend_kind=args.backend,
    )

    print(f"[driver] mode={args.mode} backend={args.backend} "
          f"cycles={args.cycles} workspace={workspace}")
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
