"""单 Domain 数据准备脚本。通过环境变量 DOMAIN 和 TARGET_TOKENS 控制。"""

import os
import time
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

OUTPUT_ROOT = Path("/fsx/dev/jiaqi/data/olmo-3b-pretrain")

DOMAIN = os.environ.get("DOMAIN", "web")
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "5000000000"))

DOMAIN_DATASETS = {
    "web": ("HuggingFaceFW/fineweb-edu", "default", "train"),
    "code": ("bigcode/starcoderdata", "default", "train"),
    "math": ("open-web-math/open-web-math", "default", "train"),
}


def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0325-32B")
    log.info(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")
    return tokenizer


def tokenize_and_save_shard(texts, tokenizer, output_path, max_seq_len=4096):
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    if not all_tokens:
        return 0

    n_tokens = (len(all_tokens) // max_seq_len) * max_seq_len
    if n_tokens == 0:
        return 0

    arr = np.array(all_tokens[:n_tokens], dtype=np.uint32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return n_tokens


def main():
    log.info(f"Domain: {DOMAIN}, Target: {TARGET_TOKENS/1e9:.1f}B tokens")

    output_dir = OUTPUT_ROOT / DOMAIN
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查已有数据，支持断点续传
    existing_shards = sorted(output_dir.glob("shard_*.npy"))
    start_shard = len(existing_shards)
    existing_tokens = 0
    for shard in existing_shards:
        arr = np.load(shard)
        existing_tokens += len(arr)
    if existing_tokens > 0:
        log.info(f"Resuming: found {len(existing_shards)} shards, {existing_tokens/1e9:.2f}B tokens")
    if existing_tokens >= TARGET_TOKENS:
        log.info("Already have enough tokens, done!")
        return

    dataset_name, subset, split = DOMAIN_DATASETS.get(DOMAIN, ("HuggingFaceFW/fineweb-edu", "default", "train"))
    log.info(f"Source: {dataset_name}")

    from datasets import load_dataset
    tokenizer = get_tokenizer()

    total_tokens = existing_tokens
    shard_idx = start_shard
    batch_size = 10000
    batch_texts = []
    t0 = time.time()

    ds = load_dataset(dataset_name, subset, split=split, streaming=True)

    for example in ds:
        text = example.get("text", example.get("content", ""))
        if not text or len(text) < 100:
            continue

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            shard_path = output_dir / f"shard_{shard_idx:05d}.npy"
            n = tokenize_and_save_shard(batch_texts, tokenizer, shard_path)
            total_tokens += n
            shard_idx += 1
            batch_texts = []

            if shard_idx % 10 == 0:
                elapsed = time.time() - t0
                speed = (total_tokens - existing_tokens) / max(elapsed, 1)
                remaining = (TARGET_TOKENS - total_tokens) / max(speed, 1)
                log.info(
                    f"[{DOMAIN}] Shard {shard_idx}: "
                    f"{total_tokens/1e9:.2f}B / {TARGET_TOKENS/1e9:.1f}B tokens "
                    f"({100*total_tokens/TARGET_TOKENS:.1f}%) "
                    f"ETA: {remaining/3600:.1f}h"
                )

            if total_tokens >= TARGET_TOKENS:
                break

    if batch_texts:
        shard_path = output_dir / f"shard_{shard_idx:05d}.npy"
        n = tokenize_and_save_shard(batch_texts, tokenizer, shard_path)
        total_tokens += n

    elapsed = time.time() - t0
    log.info(f"[{DOMAIN}] DONE: {total_tokens/1e9:.2f}B tokens in {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
