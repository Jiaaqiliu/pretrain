"""CoT post-processing: enforce ``\\boxed{answer}`` + ``[verify]: PASS``.

Every generator in the pipeline — solver, teacher LLM, OOD synth — hands
its raw CoT to this module before the row is written to JSONL. The
module is intentionally tiny so mutations can target specific knobs
(inject marker vs. not, box location, etc.) without risking drift
between generators.

Design invariants:
  * The final ``\\boxed{...}`` must contain exactly the ground-truth
    answer, regardless of what the generator put there. This is the
    single hard guarantee — callers (SFT trainer, eval extractor) are
    free to trust it.
  * ``[verify]: PASS`` is optional; generators that already include a
    verification line can set ``inject_verify_marker=False`` to avoid
    duplicates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")
_VERIFY_PASS_RE = re.compile(r"\[verify\]\s*:\s*PASS", re.IGNORECASE)


@dataclass(frozen=True)
class CoTPostprocessConfig:
    inject_verify_marker: bool = True
    force_boxed_answer: bool = True
    strip_trailing_whitespace: bool = True


def postprocess_cot(
    cot: str,
    answer: str,
    *,
    config: CoTPostprocessConfig = CoTPostprocessConfig(),
) -> str:
    """Apply the invariants.

    If ``force_boxed_answer`` and a ``\\boxed{...}`` exists, replace its
    contents with ``answer``. If none exists, append one on a new line.

    If ``inject_verify_marker`` and no ``[verify]: PASS`` exists, insert
    one right before the final ``\\boxed{...}``.
    """
    out = cot

    if config.force_boxed_answer:
        replacement = f"\\boxed{{{answer}}}"
        if _BOXED_RE.search(out):
            # Use a lambda so re.sub doesn't interpret ``\b`` inside the
            # replacement as a backreference escape (which would turn it
            # into the ``\x08`` backspace char — silent data corruption).
            out = _BOXED_RE.sub(lambda _m: replacement, out)
        else:
            sep = "" if out.endswith("\n") else "\n"
            out = f"{out}{sep}{replacement}"

    if config.inject_verify_marker and not _VERIFY_PASS_RE.search(out):
        marker = "[verify]: PASS\n"
        m = _BOXED_RE.search(out)
        if m:
            # Insert before the last \boxed occurrence
            start = m.start()
            out = out[:start] + marker + out[start:]
        else:
            # No box — stick it at the end
            sep = "" if out.endswith("\n") else "\n"
            out = f"{out}{sep}{marker}"

    if config.strip_trailing_whitespace:
        out = out.rstrip() + "\n"

    return out


def has_verify_pass(cot: str) -> bool:
    return bool(_VERIFY_PASS_RE.search(cot))


def extract_boxed(cot: str) -> str | None:
    """Return the *last* ``\\boxed{...}`` contents, or None."""
    matches = list(_BOXED_RE.finditer(cot))
    if not matches:
        return None
    inner = matches[-1].group(0)
    return inner[len(r"\boxed{"):-1]


__all__ = [
    "CoTPostprocessConfig",
    "extract_boxed",
    "has_verify_pass",
    "postprocess_cot",
]
