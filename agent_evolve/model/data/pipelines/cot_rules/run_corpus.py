"""Build the SFT training corpus from reasoning + augmentation outputs.

Produces, in the current working directory:

* ``corpus.jsonl``                          - per-entry index with metadata
* ``corpus/<problem_id>/synthetic.jsonl``   - interleaved masked/unmasked
                                              token segments

The completion for each reasoning-backed entry is::

    <reasoning text>\\n</think>\\n\\boxed{<answer>}<|im_end|>

Augmentation entries (no reasoning, no boxed) close with::

    <completion>\\n</think><|im_end|>

This is a faithful port of huikang's ``corpus.py``. Adapted to our
package layout; tokenizer / chat-template paths configurable via env
vars.

Required prerequisites in CWD::

    train.csv               # Kaggle train file with id,prompt,answer
    problems.jsonl          # output of an earlier problems-index step
    reasoning/*.txt         # output of run_reasoning.py
    augmentations/*.txt     # output of run_augmentation.py (optional)

Tokenizer paths::

    HUIKANG_TOKENIZER       # path to tokenizer.json (BPE, for completion)
    HUIKANG_CHAT_TOKENIZER  # HF tokenizer name or path (chat template);
                            # defaults to nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

Usage::

    cd <work_dir>
    python -m agent_evolve.model.data.pipelines.cot_rules.run_corpus
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

TRAIN_CSV         = Path("train.csv")
AUGMENTATIONS_DIR = Path("augmentations")
PROBLEMS_INDEX    = Path("problems.jsonl")
REASONING_DIR     = Path("reasoning")
CORPUS_DIR        = Path("corpus")
CORPUS_INDEX      = Path("corpus.jsonl")

DEFAULT_TOKENIZER_PATH = Path("tokenizer.json")
DEFAULT_CHAT_TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
TOKEN_LIMIT = 8192


def _tokenizer_path() -> Path:
    env = os.environ.get("HUIKANG_TOKENIZER")
    return Path(env) if env else DEFAULT_TOKENIZER_PATH


def _chat_tokenizer_name() -> str:
    return os.environ.get("HUIKANG_CHAT_TOKENIZER", DEFAULT_CHAT_TOKENIZER)


def load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def tokenize_prompt(
    prompt_text: str,
    chat_tokenizer,
    *,
    suffix: str = PROMPT_SUFFIX,
) -> list[int]:
    messages = [{"role": "user", "content": prompt_text + suffix}]
    return chat_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )


@dataclass
class CorpusEntry:
    problem_id: str
    category: str
    tokens: list[int]
    mask: list[int]
    masked_token_count: int
    unmasked_token_count: int
    answer: str
    included: bool = False

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def to_index_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "segment": "synthetic.jsonl",
            "category": self.category,
            "masked_token_count": self.masked_token_count,
            "unmasked_token_count": self.unmasked_token_count,
            "token_count": self.token_count,
            "answer": self.answer,
            "included": self.included,
        }


def build_segments(tokens: list[int], mask: list[int]) -> list[dict]:
    if not tokens:
        return []
    segments: list[dict] = []
    seg_start = 0
    current_type = "unmasked" if mask[0] == 1 else "masked"

    for i in range(1, len(tokens)):
        token_type = "unmasked" if mask[i] == 1 else "masked"
        if token_type != current_type:
            segments.append({
                "type":   current_type,
                "pos":    seg_start,
                "tokens": tokens[seg_start:i],
            })
            seg_start = i
            current_type = token_type

    segments.append({
        "type":   current_type,
        "pos":    seg_start,
        "tokens": tokens[seg_start:],
    })
    return segments


def run(
    work_dir: Path | str | None = None,
    *,
    tokenizer_path: Path | str | None = None,
    chat_tokenizer_name: str | None = None,
    train_csv: Path | str | None = None,
    token_limit: int | None = None,
    prompt_suffix: str | None = None,
) -> None:
    """Programmatic entry point. Use ``main()`` for CLI."""
    import os
    prev_cwd = os.getcwd()
    if work_dir is not None:
        os.chdir(work_dir)
    try:
        # Apply optional overrides via module globals (cheap, isolated).
        global TOKEN_LIMIT, PROMPT_SUFFIX
        if token_limit is not None:
            TOKEN_LIMIT = token_limit
        if prompt_suffix is not None:
            PROMPT_SUFFIX = prompt_suffix
        return _run_in_cwd(
            tokenizer_path=tokenizer_path,
            chat_tokenizer_name=chat_tokenizer_name,
            train_csv=train_csv,
        )
    finally:
        os.chdir(prev_cwd)


def _run_in_cwd(
    *,
    tokenizer_path: Path | str | None,
    chat_tokenizer_name: str | None,
    train_csv: Path | str | None,
) -> None:
    if not PROBLEMS_INDEX.exists():
        print(f"No {PROBLEMS_INDEX} found.")
        return

    tok_path = Path(tokenizer_path) if tokenizer_path else _tokenizer_path()
    if not tok_path.exists():
        print(f"FATAL: tokenizer.json not found at {tok_path}. "
              f"Set HUIKANG_TOKENIZER env var or pass tokenizer_path=.")
        return
    tokenizer = Tokenizer.from_file(str(tok_path))
    chat_tokenizer = AutoTokenizer.from_pretrained(
        chat_tokenizer_name or _chat_tokenizer_name(), trust_remote_code=True
    )

    train_csv_path = Path(train_csv) if train_csv else TRAIN_CSV
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    if train_csv_path.exists():
        with open(train_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["id"]
                prompts[pid] = row["prompt"]
                answers[pid] = row["answer"]
    else:
        print(f"warning: {train_csv_path} not found; reasoning entries will be skipped")

    problem_cats: dict[str, str] = {}
    for prob_raw in load_jsonl(PROBLEMS_INDEX):
        problem_cats[prob_raw["id"]] = prob_raw["category"]

    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True)

    entries: list[CorpusEntry] = []

    problem_ids = sorted(
        pid for pid in problem_cats
        if (REASONING_DIR / f"{pid}.txt").exists() and pid in prompts
    )

    for problem_id in problem_ids:
        category = problem_cats[problem_id]
        answer = answers[problem_id]
        reasoning_text = (REASONING_DIR / f"{problem_id}.txt").read_text().rstrip("\n")

        boxed_match = re.findall(r"\\boxed\{([^}]*)\}", reasoning_text)
        reasoning_answer = boxed_match[-1] if boxed_match else answer
        completion_text = (
            f"{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}<|im_end|>"
        )
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False).ids
        prompt_ids = tokenize_prompt(prompts[problem_id], chat_tokenizer)

        all_tokens = prompt_ids + completion_ids
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        if len(all_tokens) > TOKEN_LIMIT:
            all_tokens = all_tokens[:TOKEN_LIMIT]
            mask = mask[:TOKEN_LIMIT]

        unmasked_count = sum(mask)
        masked_count = len(mask) - unmasked_count

        entry = CorpusEntry(
            problem_id=problem_id,
            category=category,
            tokens=all_tokens,
            mask=mask,
            masked_token_count=masked_count,
            unmasked_token_count=unmasked_count,
            answer=answer,
            included=True,
        )

        segments = build_segments(all_tokens, mask)
        problem_dir = CORPUS_DIR / problem_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        with open(problem_dir / "synthetic.jsonl", "w") as f:
            for seg in segments:
                json.dump(seg, f); f.write("\n")
        entries.append(entry)

    if AUGMENTATIONS_DIR.exists():
        for aug_path in sorted(AUGMENTATIONS_DIR.glob("*.txt")):
            text = aug_path.read_text()
            category    = text.split("[category]\n", 1)[1].split("\n[prompt]\n", 1)[0]
            prompt_text = text.split("[prompt]\n", 1)[1].split("\n[completion]\n", 1)[0]
            completion  = text.split("\n[completion]\n", 1)[1].rstrip("\n")

            problem_id = aug_path.stem
            completion_text = f"{completion}\n</think><|im_end|>"
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False).ids
            prompt_ids = tokenize_prompt(prompt_text, chat_tokenizer, suffix="")

            all_tokens = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            assert len(all_tokens) <= TOKEN_LIMIT, (
                f"augmented entry {problem_id} exceeds token limit: "
                f"{len(all_tokens)} > {TOKEN_LIMIT}"
            )

            unmasked_count = sum(mask)
            masked_count = len(mask) - unmasked_count
            entry = CorpusEntry(
                problem_id=problem_id,
                category=category,
                tokens=all_tokens,
                mask=mask,
                masked_token_count=masked_count,
                unmasked_token_count=unmasked_count,
                answer=completion,
                included=True,
            )
            segments = build_segments(all_tokens, mask)
            problem_dir = CORPUS_DIR / problem_id
            problem_dir.mkdir(parents=True, exist_ok=True)
            with open(problem_dir / "synthetic.jsonl", "w") as sf:
                for seg in segments:
                    json.dump(seg, sf); sf.write("\n")
            entries.append(entry)

    entries.sort(key=lambda e: e.problem_id)
    with open(CORPUS_INDEX, "w") as f:
        for e in entries:
            json.dump(e.to_index_dict(), f); f.write("\n")

    cat_counts: dict[str, int] = {cat: 0 for cat in {e.category for e in entries}}
    cat_tokens: dict[str, int] = {cat: 0 for cat in cat_counts}
    for e in entries:
        cat_counts[e.category] += 1
        cat_tokens[e.category] += e.unmasked_token_count

    total_unmasked = sum(e.unmasked_token_count for e in entries)
    total_masked   = sum(e.masked_token_count   for e in entries)
    max_tokens     = max((e.token_count for e in entries), default=0)

    print(f"Corpus (synthetic): {len(entries)} entries")
    print(f"Unmasked tokens: {total_unmasked:,}")
    print(f"Masked tokens:   {total_masked:,}")
    print(f"Max seq length:  {max_tokens:,}")
    print()
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]} runs, {cat_tokens[cat]:,} unmasked tokens")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--chat-tokenizer-name", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--token-limit", type=int, default=None)
    args = parser.parse_args()
    run(work_dir=args.work_dir,
        tokenizer_path=args.tokenizer_path,
        chat_tokenizer_name=args.chat_tokenizer_name,
        train_csv=args.train_csv,
        token_limit=args.token_limit)


if __name__ == "__main__":
    main()
