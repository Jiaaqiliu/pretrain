"""Download and tokenize FineWeb-Edu data for real-data schedule comparison.

Tokenizes with Pythia's GPT-NeoX tokenizer, packs into 2048-length sequences,
saves as .npy shards on FSx.

Target: ~13B tokens (enough for 25K steps × 524K tokens/step)
Output: /fsx/dev/jiaqi/data/fineweb_pythia/ with shard_000.npy, shard_001.npy, ...

Each shard contains shape (N, 2048) int32 token IDs.

Usage:
    python scripts/thermo/prepare_data_fineweb.py \
        --output-dir /fsx/dev/jiaqi/data/fineweb_pythia \
        --num-tokens 14000000000 \
        --shard-size 1000000
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-tokens", type=int, default=14_000_000_000,
                        help="Total tokens to prepare (~14B for safety margin)")
    parser.add_argument("--shard-size", type=int, default=500_000,
                        help="Sequences per shard (500K × 2048 = 1B tokens/shard)")
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    eos_id = tokenizer.eos_token_id

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, eos_id={eos_id}")
    print(f"Target: {args.num_tokens / 1e9:.1f}B tokens in {args.seq_len}-length sequences")
    print(f"Output: {output_dir}")

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    token_buffer = []
    shard_idx = 0
    sequences_in_shard = []
    total_tokens = 0
    total_sequences = 0
    t0 = time.time()

    for doc in ds:
        text = doc.get("text", "")
        if not text or len(text) < 50:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        token_buffer.extend(ids)

        while len(token_buffer) >= args.seq_len:
            seq = token_buffer[:args.seq_len]
            token_buffer = token_buffer[args.seq_len:]
            sequences_in_shard.append(seq)
            total_sequences += 1
            total_tokens += args.seq_len

            if len(sequences_in_shard) >= args.shard_size:
                shard_path = output_dir / f"shard_{shard_idx:04d}.npy"
                arr = np.array(sequences_in_shard, dtype=np.int32)
                np.save(shard_path, arr)
                elapsed = time.time() - t0
                rate = total_tokens / elapsed / 1e6
                print(f"  Saved {shard_path.name}: {arr.shape} "
                      f"({total_tokens/1e9:.2f}B tokens, {rate:.1f}M tok/s)")
                sequences_in_shard = []
                shard_idx += 1

        if total_tokens >= args.num_tokens:
            break

    if sequences_in_shard:
        shard_path = output_dir / f"shard_{shard_idx:04d}.npy"
        arr = np.array(sequences_in_shard, dtype=np.int32)
        np.save(shard_path, arr)
        print(f"  Saved {shard_path.name}: {arr.shape} (final)")
        shard_idx += 1

    elapsed = time.time() - t0
    print(f"\nDone! {shard_idx} shards, {total_tokens/1e9:.2f}B tokens, {elapsed/60:.1f} min")

    meta = {
        "num_shards": shard_idx,
        "total_tokens": total_tokens,
        "total_sequences": total_sequences,
        "seq_len": args.seq_len,
        "tokenizer": "EleutherAI/pythia-410m-deduped",
        "source": "HuggingFaceFW/fineweb-edu:sample-10BT",
    }
    import json
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
