"""Verify the ``k8s_h200`` registry entry resolves to ``K8sTinkerLiteBackend``.

This doesn't instantiate the backend (that would fail without kubeconfig)
— we just check the dotted path imports cleanly and the class exists.
"""

from __future__ import annotations

import importlib


def test_k8s_dotted_path_imports() -> None:
    from agent_evolve.model.registries import TRAINING_BACKENDS

    dotted = TRAINING_BACKENDS["k8s_h200"]
    module_path, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Sanity: it's a class named K8sTinkerLiteBackend and has name="k8s_h200".
    assert cls.__name__ == "K8sTinkerLiteBackend"
    assert cls.name == "k8s_h200"


def test_single_node_registry_unchanged() -> None:
    """Regression guard: we did not alter the h200_single_node registration."""
    from agent_evolve.model.registries import TRAINING_BACKENDS

    dotted = TRAINING_BACKENDS["h200_single_node"]
    module_path, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert cls.__name__ == "SingleNodeTinkerLiteBackend"


def test_k8s_backend_subclasses_single_node() -> None:
    """Shared pipeline orchestration via inheritance — documented design."""
    from agent_evolve.backends.tinkerlite.elastic import K8sTinkerLiteBackend
    from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend

    assert issubclass(K8sTinkerLiteBackend, SingleNodeTinkerLiteBackend)
