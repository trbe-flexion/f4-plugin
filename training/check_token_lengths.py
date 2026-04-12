"""Check token lengths in training data after chat template expansion.

Loads the Llama 3.2 tokenizer, applies the chat template to each example
in the training JSONL, and prints distribution statistics. Use the output
to choose max_seq_length for training.

Requires HF access to the tokenizer. Designed to run on SageMaker.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_messages(jsonl_path: str) -> list[list[dict]]:
    """Load messages arrays from a chat-format JSONL file."""
    messages_list = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages_list.append(record["messages"])
    return messages_list


def compute_token_lengths(messages_list: list[list[dict]], tokenizer) -> list[int]:
    """Apply chat template and return token counts per example."""
    lengths = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        token_ids = tokenizer.encode(text)
        lengths.append(len(token_ids))
    return lengths


def compute_statistics(lengths: list[int]) -> dict:
    """Compute summary statistics for a list of token lengths."""
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)
    return {
        "count": n,
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": round(statistics.mean(sorted_lengths), 1),
        "median": round(statistics.median(sorted_lengths), 1),
        "p95": sorted_lengths[min(p95_idx, n - 1)],
        "p99": sorted_lengths[min(p99_idx, n - 1)],
    }


def print_statistics(stats: dict) -> None:
    """Print statistics in a readable format."""
    print(f"Examples: {stats['count']}")
    print(f"Min:      {stats['min']}")
    print(f"Max:      {stats['max']}")
    print(f"Mean:     {stats['mean']}")
    print(f"Median:   {stats['median']}")
    print(f"P95:      {stats['p95']}")
    print(f"P99:      {stats['p99']}")


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check token lengths in training data")
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help="Model ID for tokenizer",
    )
    parsed = parser.parse_args(args=[] if args is None else args)

    data_path = Path(parsed.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {parsed.model}")
    tokenizer = AutoTokenizer.from_pretrained(parsed.model)

    print(f"Loading data: {data_path}")
    messages_list = load_messages(str(data_path))

    print("Computing token lengths...")
    lengths = compute_token_lengths(messages_list, tokenizer)

    print()
    stats = compute_statistics(lengths)
    print_statistics(stats)

    if stats["p95"] <= 1024:
        print("\nRecommendation: max_seq_length=1024 covers 95%+ of examples.")
    elif stats["p95"] <= 2048:
        print("\nRecommendation: max_seq_length=2048 covers 95%+ of examples.")
    else:
        print(f"\nWarning: P95 is {stats['p95']} tokens. Consider longer max_seq_length.")


if __name__ == "__main__":
    main()
