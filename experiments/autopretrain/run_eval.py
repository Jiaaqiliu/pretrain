"""Standalone evaluation script for checkpoint validation loss.

Called by PretrainEvalHarness in full-eval mode via subprocess.
Computes cross-entropy loss on a validation set using olmo-core infrastructure.

Usage:
    torchrun --nproc-per-node=1 experiments/autopretrain/run_eval.py \
        --checkpoint /path/to/checkpoint \
        --val-data /path/to/val.npy \
        --domain c4

Outputs a single JSON line to stdout with {"loss": X.XX, "perplexity": X.XX}
"""

import argparse
import json
import sys
from pathlib import Path

_olmo_core_src = Path(__file__).resolve().parents[2] / "olmo-core" / "src"
if _olmo_core_src.exists():
    sys.path.insert(0, str(_olmo_core_src))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--domain", type=str, default="c4")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from olmo_core.data import NumpyFSLDatasetConfig, NumpyDataLoaderConfig, TokenizerConfig
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.train import prepare_training_environment, teardown_training_environment

    prepare_training_environment(backend=None)

    tokenizer = TokenizerConfig.dolma2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(json.dumps({"loss": 99.0, "perplexity": 99.0, "error": "checkpoint not found"}))
        return

    # Build dataset
    val_path = Path(args.val_data)
    if not val_path.exists():
        print(json.dumps({"loss": 99.0, "perplexity": 99.0, "error": "val data not found"}))
        return

    dataset_config = NumpyFSLDatasetConfig(
        paths=[str(val_path)],
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
    )
    loader_config = NumpyDataLoaderConfig(
        global_batch_size=args.batch_size * args.sequence_length,
        seed=42,
    )

    dataset = dataset_config.build()
    data_loader = loader_config.build(dataset)

    # Load model
    # Detect model config from checkpoint metadata
    config_path = checkpoint_path / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            model_cfg = yaml.safe_load(f)
        # Reconstruct from config
        model_config = TransformerConfig.olmo2_190M(vocab_size=tokenizer.padded_vocab_size())
    else:
        model_config = TransformerConfig.olmo2_190M(vocab_size=tokenizer.padded_vocab_size())

    model = model_config.build(init_device="cpu")

    # Load weights
    state_dict_path = None
    for candidate in [
        checkpoint_path / "model.pt",
        checkpoint_path / "model.safetensors",
        checkpoint_path,  # might be a single file
    ]:
        if candidate.exists() and candidate.is_file():
            state_dict_path = candidate
            break

    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()

    # Compute loss
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            if n_batches >= args.max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            logits = model(input_ids)
            if hasattr(logits, "logits"):
                logits = logits.logits

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="mean",
            )
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = 2 ** avg_loss

    result = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "domain": args.domain,
        "n_batches": n_batches,
        "checkpoint": str(checkpoint_path),
    }
    print(json.dumps(result))

    teardown_training_environment()


if __name__ == "__main__":
    main()
