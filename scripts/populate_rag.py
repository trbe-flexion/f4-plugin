"""Populate ChromaDB with flag passages from rag_seeds.jsonl.

Idempotent — safe to re-run. Skips existing documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.store import FlagRAGStore


def load_seeds(path: str) -> list[dict]:
    """Load records from rag_seeds.jsonl."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate ChromaDB with flag passages")
    parser.add_argument(
        "--data",
        type=str,
        default="data/rag_seeds.jsonl",
        help="Path to rag seeds JSONL file",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/chromadb",
        help="ChromaDB persistent storage directory",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)

    data_path = Path(parsed.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading seeds from: {data_path}")
    seeds = load_seeds(str(data_path))
    print(f"  Loaded {len(seeds)} records")

    print(f"Creating store at: {parsed.persist_dir}")
    store = FlagRAGStore(persist_directory=parsed.persist_dir)

    added = store.add_passages(seeds)
    total = store.count()

    print(f"  Added {added} new documents")
    print(f"  Total documents in collection: {total}")


if __name__ == "__main__":
    main()
