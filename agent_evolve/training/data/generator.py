"""``DataGenerator`` Protocol — stage-level data-production abstraction.

Today's three generation stages (``solver_distill``, ``synth_generate``
= teacher LLM, ``data_merge``) are each one ``run_*_stage`` function with
its own signature. That shape doesn't compose: a developer who wants to
add "OOD perturbation" or "self-instruct" has nowhere to plug in.

``DataGenerator`` is a uniform abstraction at the stage level. Each
generator yields ``GeneratedRow`` objects (wire format defined in
``training/data/base.py``). The registry lets stages reference a generator
by string name in ``train/pipeline.yaml``:

    - name: perturb_v1
      type: generate
      generator: rule_perturb    # ← registered via @register_data_generator

Implementations live alongside a benchmark or in a user plugin. See
``INTEGRATION.md`` §4 for a RulePerturb example.

Note: the existing ``solver_distill`` and ``synth_generate`` stage workers
continue to exist as-is — they're registered as DataGenerators *in
addition* so the new ``type: generate, generator: ...`` syntax works on
them too. No behavior change for existing pipeline YAMLs.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from .base import GeneratedRow
from .recipe import DataRecipe


@runtime_checkable
class DataGenerator(Protocol):
    """Produces training rows from a (workspace, recipe, benchmark) tuple.

    Implementations are free to be pure-Python (solver-distill),
    LLM-powered (teacher distillation), stochastic (OOD synth), or
    pull from an external source.
    """

    name: str

    def generate(
        self,
        workspace: Any,
        recipe: DataRecipe,
        *,
        benchmark: Any = None,
        budget_seconds: float | None = None,
        smoke: bool = False,
    ) -> Iterable[GeneratedRow]: ...


DATA_GENERATORS: dict[str, DataGenerator] = {}


def register_data_generator(
    name: str,
) -> Callable[[type | DataGenerator], type | DataGenerator]:
    """Decorator OR direct registration.

    Use as a decorator on a class:

        @register_data_generator("rule_perturb")
        class RulePerturbGenerator:
            name = "rule_perturb"
            def generate(self, workspace, recipe, **_):
                ...

    Or register an already-constructed instance directly:

        DATA_GENERATORS["rule_perturb"] = RulePerturbGenerator(config=...)
    """
    def _decorator(obj: type | DataGenerator) -> type | DataGenerator:
        instance = obj() if isinstance(obj, type) else obj
        if name in DATA_GENERATORS and DATA_GENERATORS[name] is not instance:
            if type(DATA_GENERATORS[name]) is not type(instance):
                raise RuntimeError(
                    f"DataGenerator {name!r} already registered "
                    f"(existing={type(DATA_GENERATORS[name]).__name__}, "
                    f"new={type(instance).__name__})."
                )
        DATA_GENERATORS[name] = instance
        return obj
    return _decorator


def resolve_data_generator(name: str) -> DataGenerator:
    gen = DATA_GENERATORS.get(name)
    if gen is None:
        raise KeyError(
            f"Unknown data generator: {name!r}. Registered: {sorted(DATA_GENERATORS)}"
        )
    return gen


__all__ = [
    "DATA_GENERATORS",
    "DataGenerator",
    "register_data_generator",
    "resolve_data_generator",
]
