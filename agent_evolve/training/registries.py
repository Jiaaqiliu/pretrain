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
}

TRAINING_ALGORITHMS: dict[str, str] = {
    "mcgs": "agent_evolve.training.algorithms.mcgs.search.MCGSSearch",
}

TRAINING_BACKENDS: dict[str, str] = {
    "h200_single_node": "agent_evolve.backends.tinkerlite.single_node.SingleNodeTinkerLiteBackend",
    "k8s_h200": "agent_evolve.backends.tinkerlite.k8s.K8sTinkerLiteBackend",
    "sklearn_backend": "agent_evolve.backends.sklearn_backend.SklearnBackend",
}


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


def resolve_backend(value: Any) -> Any:
    if isinstance(value, str):
        return _resolve(value, TRAINING_BACKENDS, "backend")
    return value
