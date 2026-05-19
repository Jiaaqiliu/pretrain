"""Concatenation augmenter: merge individually-bracketed symbols into one bracket.

Input:  【]】【}】【@】【]】
Output: 【]}@]】

Uses symbols from cryptarithm problems. Length 2-8.
Generates 1500 problems, each with 100 rows.
"""

from __future__ import annotations

import hashlib
import random

SYMBOLS = list('!"#$%&\'()*+-./:;<>?@[\\]^`{|}')

LINES_PER_PROBLEM = 100
N_PROBLEMS = 1500
DEMO_LINES = 3


def _box_individual(chars: list[str]) -> str:
    return "".join(f"【{c}】" for c in chars)


def _box_merged(chars: list[str]) -> str:
    return f"【{''.join(chars)}】"


def _random_symbols(rng: random.Random) -> list[str]:
    length = rng.randint(2, 8)
    return [rng.choice(SYMBOLS) for _ in range(length)]


def _pair(chars: list[str]) -> tuple[str, str]:
    return _box_individual(chars), _box_merged(chars)


def generate() -> list[dict[str, str]]:
    rng = random.Random(99)
    problems = []

    for i in range(N_PROBLEMS):
        demo_chars = [_random_symbols(rng) for _ in range(DEMO_LINES)]
        demo_pairs = [_pair(c) for c in demo_chars]

        sample_input_lines  = [f"{j:02d} {inp}" for j, (inp, _) in enumerate(demo_pairs)]
        sample_output_lines = [f"{j:02d} {inp} -> {out}" for j, (inp, out) in enumerate(demo_pairs)]

        test_inputs = []
        test_answers = []
        for row_num in range(LINES_PER_PROBLEM):
            inp, out = _pair(_random_symbols(rng))
            test_inputs.append(f"{row_num:02d} {inp}")
            test_answers.append(f"{row_num:02d} {inp} -> {out}")

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n"
            + "\n".join(sample_input_lines)
            + "\n\n"
            + "This is a sample output.\n"
            + "\n".join(sample_output_lines)
            + "\n\n"
            + "This is your input.\n"
            + "\n".join(test_inputs)
        )

        answer = "\n".join(test_answers)
        pid = hashlib.sha256(f"concatenation_{i}".encode()).hexdigest()[:8]
        problems.append({
            "id": pid,
            "prompt": prompt,
            "completion": answer,
            "category": "concatenation",
        })

    print(f"[concatenation] Generated {len(problems)} problems")
    return problems
