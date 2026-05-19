"""Synthetic-task augmenters for the Kaggle Nemotron-Reasoning corpus.

Each augmenter exposes::

    generate() -> list[dict]    # keys: id, prompt, completion, category

The driver ``run_augmentation.py`` walks :data:`AUGMENTERS` in order and
writes each problem to ``augmentations/<id>.txt``.

Note on order: ``matching`` reads ``reasoning/*.txt`` produced by the
reasoners pass, so the canonical pipeline is:

    1. run_reasoning.py    → reasoning/<id>.txt
    2. run_augmentation.py → augmentations/<id>.txt
    3. run_corpus.py       → corpus/<id>/*.jsonl + corpus.jsonl
"""

import importlib

# Names in canonical run order. We import lazily so that callers who
# don't need ``spelling`` (which requires the ``tokenizers`` package)
# can still use the rest.
AUGMENTER_NAMES = ("spelling", "concatenation", "splitting", "matching", "lstrip")


def _load(name: str):
    return importlib.import_module(f".{name}", __name__)


def generate_all() -> list[dict[str, str]]:
    """Run every augmenter in canonical order and concat the results."""
    out: list[dict[str, str]] = []
    for name in AUGMENTER_NAMES:
        out.extend(_load(name).generate())
    return out


__all__ = ["AUGMENTER_NAMES", "generate_all"]
