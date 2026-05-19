"""Lstrip augmenter: strip leading spaces from a bracketed symbol string.

Input:  【   $%^】
Output: 【$%^】

Uses symbols from cryptarithm problems. 1-10 symbols, 1 leading space.
Generates 300 problems, each with 100 rows.
"""

from __future__ import annotations

import hashlib
import random

SYMBOLS = list('!"#$%&\'()*+-./:;<>?@[\\]^`{|}')

LINES_PER_PROBLEM = 100
N_PROBLEMS = 300
DEMO_LINES = 3


def _box(s: str) -> str:
    return f"【{s}】"


def _random_entry(rng: random.Random) -> tuple[str, str]:
    if rng.random() < 0.5:
        length = 5
    else:
        length = rng.randint(1, 10)
    symbols = "".join(rng.choice(SYMBOLS) for _ in range(length))
    return _box(" " + symbols), _box(symbols)


def generate() -> list[dict[str, str]]:
    rng = random.Random(91)
    problems = []

    for i in range(N_PROBLEMS):
        demo_pairs = [_random_entry(rng) for _ in range(DEMO_LINES)]

        sample_input_lines  = [f"{j:02d} {inp}" for j, (inp, _) in enumerate(demo_pairs)]
        sample_output_lines = [f"{j:02d} {inp} -> {out}" for j, (inp, out) in enumerate(demo_pairs)]

        test_inputs = []
        test_answers = []
        for row_num in range(LINES_PER_PROBLEM):
            inp, out = _random_entry(rng)
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
        pid = hashlib.sha256(f"lstrip_{i}".encode()).hexdigest()[:8]
        problems.append({
            "id": pid,
            "prompt": prompt,
            "completion": answer,
            "category": "lstrip",
        })

    print(f"[lstrip] Generated {len(problems)} problems")
    return problems
