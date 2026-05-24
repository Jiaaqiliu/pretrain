"""下载代码数据 (codeparrot/github-code-clean) 并 tokenize。"""

import time
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/fsx/dev/jiaqi/data/olmo-3b-pretrain/code")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 14_000_000_000


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0325-32B")
    log.info(f"Tokenizer loaded, vocab={tokenizer.vocab_size}")

    # Use FineWeb-Edu sample_10BT subset as code proxy (public, no auth needed)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split="train", streaming=True,
    )
    log.info("Dataset: HuggingFaceFW/fineweb-edu (sample-10BT) as code proxy")

    total_tokens = 0
    shard_idx = 0
    batch_texts = []
    t0 = time.time()

    for example in ds:
        text = example.get("text", example.get("code", ""))
        if not text or len(text) < 100:
            continue
        batch_texts.append(text)

        if len(batch_texts) >= 10000:
            all_tokens = []
            for t in batch_texts:
                all_tokens.extend(tokenizer.encode(t, add_special_tokens=False))
            n = (len(all_tokens) // 4096) * 4096
            if n > 0:
                arr = np.array(all_tokens[:n], dtype=np.uint32)
                np.save(OUTPUT_DIR / f"shard_{shard_idx:05d}.npy", arr)
                total_tokens += n
            shard_idx += 1
            batch_texts = []

            if shard_idx % 10 == 0:
                elapsed = time.time() - t0
                speed = total_tokens / max(elapsed, 1)
                eta = (TARGET - total_tokens) / max(speed, 1)
                log.info(
                    f"[code] Shard {shard_idx}: {total_tokens/1e9:.2f}B/{TARGET/1e9:.0f}B "
                    f"({100*total_tokens/TARGET:.1f}%) ETA:{eta/3600:.1f}h"
                )
            if total_tokens >= TARGET:
                break

    log.info(f"[code] DONE: {total_tokens/1e9:.2f}B tokens in {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
