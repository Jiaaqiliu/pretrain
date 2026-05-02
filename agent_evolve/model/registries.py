"""Registries for TrainingEvolver string-key resolution.

Mirrors the dotted-path registry pattern used by `agent_evolve.api`.
"""

from __future__ import annotations

import importlib
from typing import Any

from .types import TrainingRegistryError


TRAINING_BENCHMARKS: dict[str, str] = {
    "nemo_reasoner": "agent_evolve.benchmarks.nemo_reasoner.NemoReasonerBenchmark",
    "mle_bench": "agent_evolve.benchmarks.mle_bench.mle_bench.MLEBenchAdapter",
    "posttrain_bench": "agent_evolve.benchmarks.posttrain_bench.PosttrainBenchmark",
}

TRAINING_ALGORITHMS: dict[str, str] = {
    "mcgs": "agent_evolve.model.algorithms.mcgs.search.MCGSSearch",
    # Orchestrator-worker MAS — independent of mcgs; same workspace contract.
    # See seed_workspaces/nemo_mas_reasoner/DESIGN.md.
    "nemo_mas": "agent_evolve.model.algorithms.nemo_mas.orchestrator.NemoMASAlgorithm",
}

# The authoritative name is ``TRAINING_JOB_RUNNERS`` (implements
# ``TrainingJobRunner`` Protocol in ``runner_protocol.py``). A registered
# value can be a non-LLM job (sklearn, JAX) as easily as the default LLM
# backends. ``TRAINING_BACKENDS`` is kept as a backward-compat alias that
# mutates through the same dict.
TRAINING_JOB_RUNNERS: dict[str, str] = {
    "h200_single_node": "agent_evolve.backends.tinkerlite.single_node.SingleNodeTinkerLiteBackend",
    "k8s_h200": "agent_evolve.backends.tinkerlite.elastic.K8sTinkerLiteBackend",
    # Non-LLM runner — implements ``TrainingJobRunner`` directly. See
    # ``INTEGRATION.md §1`` (sklearn / tabular AutoML).
    "sklearn_backend": "agent_evolve.backends.sklearn_backend.SklearnBackend",
}
TRAINING_BACKENDS = TRAINING_JOB_RUNNERS  # deprecated alias; use TRAINING_JOB_RUNNERS


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _resolve(name: str, registry: dict[str, str], kind: str) -> Any:
    dotted = registry.get(name)
    if not dotted:
        raise TrainingRegistryError(
            f"Unknown {kind}: {name!r}. Available: {sorted(registry)}"
        )
    try:
        cls = _import_class(dotted)
    except (ImportError, AttributeError) as exc:
        raise TrainingRegistryError(
            f"Failed to import {kind} {name!r} from {dotted}: {exc}"
        ) from exc
    return cls()


def resolve_benchmark(value: Any) -> Any:
    if isinstance(value, str):
        return _resolve(value, TRAINING_BENCHMARKS, "benchmark")
    return value


def resolve_algorithm(value: Any) -> Any:
    if isinstance(value, str):
        return _resolve(value, TRAINING_ALGORITHMS, "algorithm")
    return value


def resolve_job_runner(value: Any) -> Any:
    if isinstance(value, str):
        return _resolve(value, TRAINING_JOB_RUNNERS, "job_runner")
    return value


# Backward-compat alias. New code should call ``resolve_job_runner``.
resolve_backend = resolve_job_runner
