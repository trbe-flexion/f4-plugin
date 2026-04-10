"""Carve out a stratified test set from train.jsonl -> test.jsonl.

Samples 8 examples per flag and 50 no_flag examples, removes them from
train.jsonl, and writes them to data/test.jsonl.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
PER_FLAG = 8
NO_FLAG_COUNT = 50
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"


def get_labels(record: dict) -> list[str]:
    assistant_msg = next(m for m in record["messages"] if m["role"] == "assistant")
    return [f.strip() for f in assistant_msg["content"].strip().splitlines()]


def main() -> None:
    random.seed(SEED)

    with TRAIN_PATH.open() as f:
        records = [json.loads(line) for line in f]

    # Group indices by primary label (first flag listed, or no_flag)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        labels = get_labels(record)
        primary = labels[0]
        buckets[primary].append(i)

    test_indices: set[int] = set()

    for label, indices in buckets.items():
        n = NO_FLAG_COUNT if label == "no_flag" else PER_FLAG
        sample = random.sample(indices, min(n, len(indices)))
        test_indices.update(sample)

    test_records = [records[i] for i in sorted(test_indices)]
    train_records = [r for i, r in enumerate(records) if i not in test_indices]

    with TEST_PATH.open("w") as f:
        for record in test_records:
            f.write(json.dumps(record) + "\n")

    with TRAIN_PATH.open("w") as f:
        for record in train_records:
            f.write(json.dumps(record) + "\n")

    print(f"Test set:  {len(test_records)} records -> {TEST_PATH}")
    print(f"Train set: {len(train_records)} records -> {TRAIN_PATH}")


if __name__ == "__main__":
    main()
