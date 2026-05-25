"""Run downstream benchmarks on saved checkpoints using lm-evaluation-harness.

Evaluates checkpoints from the 3-way schedule comparison on standard NLP benchmarks.

Prerequisites:
    pip install lm-eval>=0.4.0

Usage:
    # Single checkpoint
    python scripts/eval/run_benchmarks.py \
        --checkpoint /path/to/checkpoint \
        --tokenizer EleutherAI/pythia-1b-deduped \
        --output results/eval/1b_cosine_final.json

    # Batch mode: all checkpoints in a directory
    python scripts/eval/run_benchmarks.py \
        --checkpoint-dir /path/to/ckpts/ \
        --tokenizer EleutherAI/pythia-1b-deduped \
        --output-dir results/eval/1b_cosine/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


TASKS = [
    "lambada_openai",
    "piqa",
    "winogrande",
    "arc_easy",
    "hellaswag",
]

TASK_STRING = ",".join(TASKS)


def run_eval(checkpoint_path: str, tokenizer: str, output_path: str,
             batch_size: int = 32, num_fewshot: int = 0):
    """Run lm-eval-harness on a single checkpoint."""
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint_path},tokenizer={tokenizer}",
        "--tasks", TASK_STRING,
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", output_path,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: lm-eval failed for {checkpoint_path}")
        print(result.stderr[-2000:] if result.stderr else "No stderr")
        return None

    print(result.stdout[-1000:] if result.stdout else "")
    return output_path


def parse_results(output_dir: str) -> dict:
    """Parse lm-eval output JSONs into a summary."""
    output_path = Path(output_dir)
    results = {}

    for json_file in output_path.rglob("results.json"):
        with open(json_file) as f:
            data = json.load(f)

        if "results" in data:
            for task_name, task_results in data["results"].items():
                metric_key = "acc,none" if "acc,none" in task_results else "acc_norm,none"
                if metric_key in task_results:
                    results[task_name] = task_results[metric_key]

    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Single checkpoint path")
    group.add_argument("--checkpoint-dir", help="Directory containing multiple checkpoints")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name/path")
    parser.add_argument("--output", help="Output path for single checkpoint")
    parser.add_argument("--output-dir", help="Output directory for batch mode")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-fewshot", type=int, default=0)
    args = parser.parse_args()

    if args.checkpoint:
        output = args.output or f"results/eval/{Path(args.checkpoint).name}.json"
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        run_eval(args.checkpoint, args.tokenizer, output, args.batch_size, args.num_fewshot)

        results = parse_results(Path(output).parent)
        if results:
            print(f"\n=== Benchmark Results ===")
            total = 0
            for task, score in sorted(results.items()):
                print(f"  {task}: {score:.4f}")
                total += score
            print(f"  Average: {total/len(results):.4f}")
    else:
        ckpt_dir = Path(args.checkpoint_dir)
        output_dir = Path(args.output_dir or f"results/eval/{ckpt_dir.name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])
        print(f"Found {len(checkpoints)} checkpoints in {ckpt_dir}")

        all_results = {}
        for ckpt in checkpoints:
            print(f"\n--- Evaluating: {ckpt.name} ---")
            out_path = str(output_dir / ckpt.name)
            run_eval(str(ckpt), args.tokenizer, out_path, args.batch_size, args.num_fewshot)
            results = parse_results(out_path)
            if results:
                all_results[ckpt.name] = results
                avg = sum(results.values()) / len(results)
                print(f"  Average: {avg:.4f}")

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
