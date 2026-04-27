"""PR1 invariant: existing ae.Evolver is not touched by the new training code."""

from __future__ import annotations


def test_ae_Evolver_still_importable() -> None:
    import agent_evolve as ae

    assert hasattr(ae, "Evolver")
    assert hasattr(ae, "TrainingEvolver")
    assert ae.Evolver.__module__ == "agent_evolve.api"


def test_ae_Evolver_signature_unchanged() -> None:
    import inspect

    import agent_evolve as ae

    sig = inspect.signature(ae.Evolver.__init__)
    # Match the signature in agent_evolve/api.py
    expected_params = ["self", "agent", "benchmark", "config", "engine", "work_dir"]
    assert list(sig.parameters) == expected_params
