"""Default deterministic reasoners (solvers) for the Kaggle
Nemotron-Reasoning domains.

Each domain module exposes a single uniform entry point::

    solve(problem: Problem) -> str | None

returning a CoT trace ending in ``\\boxed{answer}``. The :data:`SOLVERS`
map is keyed by Kaggle domain name::

    from agent_evolve.model.data.reasoners import SOLVERS, DOMAINS
    trace = SOLVERS["bits"](problem)

Kaggle domain → file → entry point:

  bits      → bits.py      → solve
  cipher    → cipher.py    → solve
  gravity   → gravity.py   → solve
  numerals  → numerals.py  → solve
  units     → units.py     → solve
  equations → equation_numeric.py → reasoning_equation_numeric
                                    (cryptarithm.py is a partial
                                    sub-strategy not in the dispatcher)
"""

from .bits import solve as _bits_solve
from .cipher import solve as _cipher_solve
from .cryptarithm import reasoning_cryptarithm
from .equation_numeric import reasoning_equation_numeric
from .gravity import solve as _gravity_solve
from .numerals import solve as _numerals_solve
from .store_types import Example, Problem, ProblemCategory
from .units import solve as _units_solve

SOLVERS = {
    "bits":      _bits_solve,
    "cipher":    _cipher_solve,
    "equations": reasoning_equation_numeric,
    "gravity":   _gravity_solve,
    "numerals":  _numerals_solve,
    "units":     _units_solve,
}

DOMAINS = tuple(SOLVERS.keys())


def get_solver(domain: str):
    """Return the default reasoner for a Kaggle domain, or None if unknown."""
    return SOLVERS.get(domain)


__all__ = [
    "SOLVERS",
    "DOMAINS",
    "get_solver",
    "Example",
    "Problem",
    "ProblemCategory",
    "reasoning_cryptarithm",
    "reasoning_equation_numeric",
]
