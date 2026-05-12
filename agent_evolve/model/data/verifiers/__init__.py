"""Label verifiers for the Nemotron Kaggle-style reasoning domains.

This subpackage groups the tooling used to audit ``balanced_dev*`` splits
and the full Kaggle ``train.csv`` pool.

Modules (CLI / programmatic):

* :mod:`.bits` — 8-bit rule-family consistency checker
  (``is_bits_label_consistent``; CLI: ``python -m
  agent_evolve.model.data.verifiers.bits``).
* :mod:`.equations` — S1–S5 symbolic equation verifier
  (``solve_row``; CLI: ``python -m
  agent_evolve.model.data.verifiers.equations``). Arithmetic-only fast path
  available via :mod:`.equations_arith`.
* :mod:`.programmatic` — huikang ``reasoning_*`` solver suite as a ground-
  truth oracle over all 6 Kaggle-scored domains.
* :mod:`.opus_judge` — Claude Opus 4.6 judgements on labels (Bedrock).
* :mod:`.scan_kaggle` — unified scanner over the full Kaggle ``train.csv``
  that dispatches each row to the strongest verifier for its domain.

The huikang solver ports used as oracles live in
:mod:`agent_evolve.model.data.reasoners`.

Importing this package does NOT eagerly import submodules; do
``from .bits import is_bits_label_consistent`` etc. explicitly.
"""

__all__ = [
    "bits",
    "equations",
    "equations_arith",
    "opus_judge",
    "programmatic",
    "scan_kaggle",
]
