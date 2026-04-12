"""Re-wrap training data with realistic RAG context.

Replaces the simulated RAG context (which always used the correct flag's seeds)
with actual ChromaDB retrieval results (which may or may not be relevant).
This fixes the training/inference mismatch identified during Bedrock testing.

Reads existing train.jsonl and eval.jsonl, preserves chunks and labels,
only changes the RAG context portion of the user message.

Usage:
    PYTHONPATH=. uv run python scripts/rewrap_rag_context.py
    PYTHONPATH=. uv run python scripts/rewrap_rag_context.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.rag.retriever import format_context
from src.rag.store import FlagRAGStore

DATA_DIR = Path("data")
TOP_K = 3


def extract_chunk(user_content: str) -> str:
    """Extract the RFP chunk from a training user message.

    Training format is:
        [Retrieved context]
        ...
        ---

        RFP chunk to analyze:
        <chunk text>
    """
    marker = "RFP chunk to analyze:\n"
    if marker in user_content:
        return user_content.split(marker, 1)[1]

    # Fallback: if no marker, the whole message is the chunk
    return user_content


def rewrap_file(
    input_path: Path,
    output_path: Path,
    store: FlagRAGStore,
    top_k: int,
) -> int:
    """Rewrap a JSONL file with real RAG context. Returns record count."""
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rewrapped = []
    for record in records:
        messages = record["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        user_msg = next(m for m in messages if m["role"] == "user")
        assistant_msg = next(m for m in messages if m["role"] == "assistant")

        chunk = extract_chunk(user_msg["content"])
        results = store.query(chunk, top_k=top_k)

        new_user_content = format_context(results, chunk) if results else chunk

        rewrapped.append(
            {
                "messages": [
                    system_msg,
                    {"role": "user", "content": new_user_content},
                    assistant_msg,
                ]
            }
        )

    with open(output_path, "w") as f:
        for record in rewrapped:
            f.write(json.dumps(record) + "\n")

    return len(rewrapped)


def main():
    parser = argparse.ArgumentParser(description="Re-wrap training data with realistic RAG context")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    train_path = DATA_DIR / "train.jsonl"
    eval_path = DATA_DIR / "eval.jsonl"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError("train.jsonl and eval.jsonl must exist in data/")

    print("Populating ChromaDB from rag_seeds.jsonl...")
    store = FlagRAGStore()  # ephemeral, no persist
    seeds = []
    with open(DATA_DIR / "rag_seeds.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    added = store.add_passages(seeds)
    print(f"  {added} passages loaded")

    if args.dry_run:
        # Just show stats
        for path in [train_path, eval_path]:
            with open(path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            user = next(m["content"] for m in records[0]["messages"] if m["role"] == "user")
            chunk = extract_chunk(user)
            results = store.query(chunk, top_k=args.top_k)
            print(f"\n{path.name}: {len(records)} records")
            print(f"  Sample chunk: {chunk[:100]}...")
            print(f"  RAG results: {[r['flag'] for r in results]}")
        return

    # Backup originals
    backup_dir = DATA_DIR / "backup_v1"
    backup_dir.mkdir(exist_ok=True)
    for path in [train_path, eval_path]:
        backup = backup_dir / path.name
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"  Backed up {path.name} → {backup}")
        else:
            print(f"  Backup already exists: {backup}")

    print(f"\nRe-wrapping with top_k={args.top_k}...")

    n_train = rewrap_file(train_path, train_path, store, args.top_k)
    print(f"  train.jsonl: {n_train} records")

    n_eval = rewrap_file(eval_path, eval_path, store, args.top_k)
    print(f"  eval.jsonl: {n_eval} records")

    # Also rewrap test.jsonl if it exists
    test_path = DATA_DIR / "test.jsonl"
    if test_path.exists():
        backup = backup_dir / test_path.name
        if not backup.exists():
            shutil.copy2(test_path, backup)
        n_test = rewrap_file(test_path, test_path, store, args.top_k)
        print(f"  test.jsonl: {n_test} records")

    print("\nDone. Originals backed up to data/backup_v1/")


if __name__ == "__main__":
    main()
