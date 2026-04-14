#!/usr/bin/env python3
"""Convert labeled_real.jsonl into RAG seeds + train/eval/test splits.

Reads the raw Claude-labeled chunks, selects RAG seeds from high-confidence
examples, builds ChromaDB from combined synthetic + real seeds, wraps each
remaining chunk with retrieved RAG context, and writes the final training files.

Usage:
  PYTHONPATH=. uv run python scripts/build_training_set.py
  PYTHONPATH=. uv run python scripts/build_training_set.py --rag-per-flag 10 --split 80/10/10
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.rag.retriever import format_context
from src.rag.store import FlagRAGStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
SYSTEM_PROMPT_PATH = REPO_ROOT / "scripts" / "system-prompt.md"

TOP_K = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(records)} records → {path.relative_to(REPO_ROOT)}")


def make_training_record(
    system_prompt: str,
    chunk_text: str,
    flags: list[str],
    rag_context: list[dict],
) -> dict:
    """Build a chat-messages training record."""
    user_content = format_context(rag_context, chunk_text) if rag_context else chunk_text
    label = "\n".join(flags) if flags else "no_flag"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label},
        ]
    }


# ---------------------------------------------------------------------------
# RAG seed selection
# ---------------------------------------------------------------------------


def select_rag_seeds(
    chunks: list[dict],
    per_flag: int,
) -> tuple[list[dict], list[dict]]:
    """Select RAG seeds from high-confidence, non-adversarial chunks.

    Returns (rag_seeds, remaining_chunks).
    Seeds are split: half go to RAG, half stay in training pool.
    """
    # Group flagged, high-confidence, non-adversarial chunks by flag
    candidates: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        if chunk.get("adversarial"):
            continue
        if chunk.get("confidence") != "high":
            continue
        for flag in chunk.get("flags", []):
            candidates[flag].append(chunk)

    # Select seeds — take top candidates per flag
    seed_ids: set[int] = set()
    rag_seed_records: list[dict] = []

    for flag, flag_chunks in candidates.items():
        # Deduplicate by chunk id
        unseen = [c for c in flag_chunks if id(c) not in seed_ids]
        selected = unseen[: per_flag * 2]  # take 2x, split half to RAG / half to training

        for i, chunk in enumerate(selected):
            if i < per_flag:
                # Goes to RAG seeds
                rag_seed_records.append({"flag": flag, "passage": chunk["chunk_text"]})
            seed_ids.add(id(chunk))

    # Remaining = everything not exclusively used as a RAG seed
    # Chunks used as RAG seeds but also kept for training (the second half) stay in remaining
    rag_only_ids = set()
    flag_counters: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        chunk_id = id(chunk)
        if chunk_id in seed_ids:
            # Check if this was in the RAG half (first per_flag) for any flag
            dominated_by_rag = False
            for flag in chunk.get("flags", []):
                if flag_counters[flag] < per_flag:
                    dominated_by_rag = True
                flag_counters[flag] += 1
            if dominated_by_rag:
                rag_only_ids.add(chunk_id)

    remaining = [c for c in chunks if id(c) not in rag_only_ids]

    return rag_seed_records, remaining


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training set from labeled real data")
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "labeled_real.jsonl"),
        help="Path to labeled_real.jsonl",
    )
    parser.add_argument("--rag-per-flag", type=int, default=10)
    parser.add_argument("--split", default="80/10/10", help="Train/eval/test split percentages")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-synthetic-seeds",
        action="store_true",
        default=True,
        help="Include synthetic rag_seeds.jsonl in ChromaDB (default: true)",
    )
    parser.add_argument(
        "--no-synthetic-seeds",
        action="store_true",
        help="Exclude synthetic seeds from ChromaDB",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    include_synthetic = args.include_synthetic_seeds and not args.no_synthetic_seeds

    # Parse split
    parts = args.split.split("/")
    if len(parts) != 3:
        raise ValueError(f"Split must be train/eval/test like 80/10/10, got: {args.split}")
    train_pct, eval_pct, test_pct = (int(p) for p in parts)
    total_pct = train_pct + eval_pct + test_pct
    if total_pct != 100:
        raise ValueError(f"Split percentages must sum to 100, got {total_pct}")

    # Load labeled data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    raw_chunks = load_jsonl(input_path)

    # Deduplicate by chunk_text (overlapping sections can produce duplicates)
    seen_texts: set[str] = set()
    chunks: list[dict] = []
    for c in raw_chunks:
        text = c["chunk_text"]
        if text not in seen_texts:
            seen_texts.add(text)
            chunks.append(c)
    dupes = len(raw_chunks) - len(chunks)
    print(f"Loaded {len(raw_chunks)} labeled chunks from {input_path}")
    if dupes:
        print(f"  Removed {dupes} duplicate chunks")

    # Stats
    flagged = sum(1 for c in chunks if c.get("flags"))
    no_flag = sum(1 for c in chunks if not c.get("flags"))
    adversarial = sum(1 for c in chunks if c.get("adversarial"))
    print(f"  Flagged: {flagged} | No-flag: {no_flag} | Adversarial: {adversarial}")

    # Load system prompt
    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()

    # Step 1: Select RAG seeds
    print("\n=== Step 1: RAG seed selection ===")
    rag_seeds, remaining = select_rag_seeds(chunks, args.rag_per_flag)
    print(f"  RAG seeds: {len(rag_seeds)}")
    print(f"  Remaining for training: {len(remaining)}")

    # Count seeds per flag
    seed_flags: dict[str, int] = defaultdict(int)
    for s in rag_seeds:
        seed_flags[s["flag"]] += 1
    for flag, count in sorted(seed_flags.items()):
        print(f"    {flag}: {count}")

    # Write RAG seeds
    rag_seeds_path = DATA_DIR / "rag_seeds_real.jsonl"
    write_jsonl(rag_seeds_path, rag_seeds)

    # Step 2: Build ChromaDB
    print("\n=== Step 2: Build ChromaDB ===")
    store = FlagRAGStore()  # ephemeral, no persist

    # Load synthetic seeds if requested
    synthetic_seeds_path = DATA_DIR / "synthetic" / "rag_seeds.jsonl"
    if include_synthetic and synthetic_seeds_path.exists():
        synthetic_seeds = load_jsonl(synthetic_seeds_path)
        added = store.add_passages(synthetic_seeds)
        print(f"  Added {added} synthetic seeds")

    # Add real seeds
    added = store.add_passages(rag_seeds)
    print(f"  Added {added} real seeds")
    print(f"  Total in ChromaDB: {store.count()}")

    # Step 3: Wrap with RAG context and build training records
    print("\n=== Step 3: Wrap with RAG context ===")
    training_records: list[dict] = []
    for chunk in remaining:
        chunk_text = chunk["chunk_text"]
        flags = chunk.get("flags", [])

        # Query ChromaDB, excluding self (by checking for exact match)
        results = store.query(chunk_text, top_k=args.top_k + 1)
        # Filter out exact self-matches
        results = [r for r in results if r["passage"] != chunk_text][: args.top_k]

        record = make_training_record(system_prompt, chunk_text, flags, results)
        training_records.append(record)

    print(f"  Built {len(training_records)} training records")

    # Step 4: Shuffle and split
    print("\n=== Step 4: Split ===")
    random.shuffle(training_records)

    n = len(training_records)
    train_end = int(n * train_pct / 100)
    eval_end = train_end + int(n * eval_pct / 100)

    train_set = training_records[:train_end]
    eval_set = training_records[train_end:eval_end]
    test_set = training_records[eval_end:]

    write_jsonl(DATA_DIR / "train.jsonl", train_set)
    write_jsonl(DATA_DIR / "eval.jsonl", eval_set)
    write_jsonl(DATA_DIR / "test.jsonl", test_set)

    print(f"\nDone. Train: {len(train_set)} | Eval: {len(eval_set)} | Test: {len(test_set)}")


if __name__ == "__main__":
    main()
