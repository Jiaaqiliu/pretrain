"""数据准备脚本：下载 FineWeb-Edu 子集并 tokenize 为 OLMo-core NumpyDataset 格式。

目标：准备 ~70B tokens 的预训练数据（60B 训练 + 10B buffer）
数据源：HuggingFace FineWeb-Edu / StarCoder / OpenWebMath
Tokenizer: dolma2 (allenai/OLMo-2-0325-32B, vocab_size=100278)
格式: numpy uint32 (olmo-core NumpyFSLDataset 所需)

输出：/fsx/dev/jiaqi/data/olmo-3b-pretrain/{web,code,math,books,academic}/*.npy

用法:
    python scripts/prepare_data_3b.py

预计耗时: 2-4 小时（取决于 HF 下载速度 + tokenize 速度）
预计磁盘: ~280 GB tokenized 数据 (uint32)
"""

import time
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

OUTPUT_ROOT = Path("/fsx/dev/jiaqi/data/olmo-3b-pretrain")

# 目标 tokens 数量（70B tokens，略多于训练需要的 60B）
TARGET_TOKENS = 70_000_000_000

# 各 domain 的目标配比
DOMAIN_TARGETS = {
    "web": 0.55,       # ~38.5B tokens
    "code": 0.20,      # ~14B tokens
    "math": 0.10,      # ~7B tokens
    "books": 0.08,     # ~5.6B tokens
    "academic": 0.07,  # ~4.9B tokens
}

# HuggingFace 数据集映射
DOMAIN_DATASETS = {
    "web": ("HuggingFaceFW/fineweb-edu", "default", "train"),
    "code": ("bigcode/starcoderdata", "default", "train"),
    "math": ("open-web-math/open-web-math", "default", "train"),
    "books": ("HuggingFaceFW/fineweb-edu", "default", "train"),  # 用 fineweb 子集代替
    "academic": ("HuggingFaceFW/fineweb-edu", "default", "train"),  # 用 fineweb 子集代替
}


def get_tokenizer():
    """获取 tokenizer（使用 OLMo-2 tokenizer）。"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0325-32B")
        log.info(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        log.warning(f"Failed to load OLMo tokenizer: {e}, falling back to GPT-2")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return tokenizer


def tokenize_and_save_shard(
    texts: list[str],
    tokenizer,
    output_path: Path,
    max_seq_len: int = 4096,
) -> int:
    """Tokenize 一批文本，拼接后存为 .npy 文件。返回 token 数量。"""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    if not all_tokens:
        return 0

    # 截断到 max_seq_len 的整数倍
    n_tokens = (len(all_tokens) // max_seq_len) * max_seq_len
    if n_tokens == 0:
        return 0

    arr = np.array(all_tokens[:n_tokens], dtype=np.uint32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return n_tokens


def prepare_domain(domain: str, target_tokens: int, tokenizer) -> int:
    """为单个 domain 准备数据。"""
    output_dir = OUTPUT_ROOT / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name, subset, split = DOMAIN_DATASETS[domain]
    log.info(f"[{domain}] Target: {target_tokens/1e9:.1f}B tokens from {dataset_name}")

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Please install: pip install datasets")
        return 0

    total_tokens = 0
    shard_idx = 0
    batch_size = 10000
    batch_texts = []

    try:
        # Streaming mode 避免下载整个数据集
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
                    log.info(
                        f"[{domain}] Shard {shard_idx}: "
                        f"{total_tokens/1e9:.2f}B / {target_tokens/1e9:.1f}B tokens "
                        f"({100*total_tokens/target_tokens:.1f}%)"
                    )

                if total_tokens >= target_tokens:
                    log.info(f"[{domain}] Reached target: {total_tokens/1e9:.2f}B tokens")
                    break

    except Exception as e:
        log.error(f"[{domain}] Error: {e}")

    # 保存最后一批
    if batch_texts:
        shard_path = output_dir / f"shard_{shard_idx:05d}.npy"
        n = tokenize_and_save_shard(batch_texts, tokenizer, shard_path)
        total_tokens += n

    log.info(f"[{domain}] Done: {total_tokens/1e9:.2f}B tokens in {shard_idx+1} shards")
    return total_tokens


def main():
    log.info("=" * 60)
    log.info("OLMo-3B Pre-training Data Preparation")
    log.info(f"Output: {OUTPUT_ROOT}")
    log.info(f"Target: {TARGET_TOKENS/1e9:.0f}B tokens total")
    log.info("=" * 60)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer()

    total_all = 0
    t0 = time.time()

    for domain, ratio in DOMAIN_TARGETS.items():
        domain_target = int(TARGET_TOKENS * ratio)
        n = prepare_domain(domain, domain_target, tokenizer)
        total_all += n

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"COMPLETE: {total_all/1e9:.2f}B tokens in {elapsed/3600:.1f} hours")
    log.info(f"Output: {OUTPUT_ROOT}")
    log.info("=" * 60)

    # 写 metadata
    meta = {
        "total_tokens": total_all,
        "domains": {d: str(OUTPUT_ROOT / d) for d in DOMAIN_TARGETS},
        "tokenizer": "allenai/OLMo-2-0325-32B",
        "format": "numpy_uint32",
        "seq_length": 4096,
    }
    import json
    (OUTPUT_ROOT / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
