#!/usr/bin/env python3
"""Build RAG seeds + train/eval/test splits from opus_validated_real.jsonl.

Steps:
  1. Load usable (non-dropped) records
  2. Reserve 2 RAG exemplars per flag (excluded from all splits)
  3. Cap negatives at 1:1 ratio with positives
  4. Split remaining by rfp_id into 80/10/10 train/eval/test
  5. Wrap each chunk with ChromaDB RAG context
  6. Write output files

Usage:
  PYTHONPATH=. uv run python scripts/build_training_set.py
  PYTHONPATH=. uv run python scripts/build_training_set.py --dry-run
  PYTHONPATH=. uv run python scripts/build_training_set.py --split 85/10/5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.rag.retriever import format_context
from src.rag.store import FlagRAGStore

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
SYSTEM_PROMPT_PATH = REPO_ROOT / "scripts" / "system-prompt.md"

TOP_K = 3
RAG_PER_FLAG = 2
FLAG_CAP: dict[str, int] = {
    "off_the_shelf_software": 100,
}


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
    records: list[dict],
    per_flag: int,
) -> tuple[list[dict], list[dict]]:
    """Pick per_flag exemplars per flag for RAG. Prefer high-confidence,
    non-adversarial, non-synthetic. Returns (rag_seeds, remaining)."""

    # Group candidates by flag
    candidates: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("adversarial"):
            continue
        for flag in rec.get("flags", []):
            candidates[flag].append(rec)

    # Sort each flag's candidates: high confidence first, then non-synthetic
    def sort_key(rec: dict) -> tuple[int, int]:
        conf = 0 if rec.get("confidence") == "high" else 1
        synth = 1 if rec.get("confidence") == "synthetic" else 0
        return (conf, synth)

    reserved_ids: set[int] = set()
    rag_seeds: list[dict] = []

    for flag, flag_recs in candidates.items():
        # Sort and pick unseen
        flag_recs.sort(key=sort_key)
        picked = 0
        for rec in flag_recs:
            if id(rec) in reserved_ids:
                continue
            rag_seeds.append({"flag": flag, "passage": rec["chunk_text"]})
            reserved_ids.add(id(rec))
            picked += 1
            if picked >= per_flag:
                break

    remaining = [r for r in records if id(r) not in reserved_ids]
    return rag_seeds, remaining


# ---------------------------------------------------------------------------
# Negative capping
# ---------------------------------------------------------------------------


def cap_negatives(records: list[dict], ratio: float = 1.0) -> list[dict]:
    """Downsample no-flag records to ratio * num_positive."""
    positives = [r for r in records if r.get("flags")]
    negatives = [r for r in records if not r.get("flags")]
    cap = int(len(positives) * ratio)

    if len(negatives) <= cap:
        print(f"  Negatives ({len(negatives)}) already within cap ({cap})")
        return records

    random.shuffle(negatives)
    kept_negatives = negatives[:cap]
    print(
        f"  Capped negatives: {len(negatives)} → {len(kept_negatives)} "
        f"(ratio {ratio}:1 with {len(positives)} positives)"
    )
    return positives + kept_negatives


# ---------------------------------------------------------------------------
# Per-flag capping
# ---------------------------------------------------------------------------


def cap_flags(records: list[dict], caps: dict[str, int]) -> list[dict]:
    """Downsample records for flags that exceed their cap."""
    if not caps:
        return records

    # Count per flag
    flag_records: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        for flag in rec.get("flags", []):
            if flag in caps:
                flag_records[flag].append(i)

    drop_indices: set[int] = set()
    for flag, cap in caps.items():
        indices = flag_records.get(flag, [])
        if len(indices) <= cap:
            continue
        random.shuffle(indices)
        to_drop = indices[cap:]
        drop_indices.update(to_drop)
        print(f"  Capped {flag}: {len(indices)} → {cap} (dropping {len(to_drop)} records)")

    if not drop_indices:
        print("  No flags exceeded caps")
        return records

    return [r for i, r in enumerate(records) if i not in drop_indices]


# ---------------------------------------------------------------------------
# Stratified split by rfp_id
# ---------------------------------------------------------------------------


def split_by_rfp_id(
    records: list[dict],
    train_pct: int,
    eval_pct: int,
    test_pct: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by rfp_id: place rare-flag RFPs first to ensure
    proportional flag coverage, then fill remaining RFPs by count."""

    # Group records by rfp_id
    by_rfp: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_rfp[rec["rfp_id"]].append(rec)

    # For each rfp_id, collect its flags
    rfp_flags: dict[str, set[str]] = {}
    for rfp_id, recs in by_rfp.items():
        flags: set[str] = set()
        for r in recs:
            flags.update(r.get("flags", []))
        rfp_flags[rfp_id] = flags

    # Count total per flag to determine rarity order
    flag_totals: Counter = Counter()
    for flags in rfp_flags.values():
        flag_totals.update(flags)

    # Sort flags rarest-first
    flags_by_rarity = [f for f, _ in flag_totals.most_common()]
    flags_by_rarity.reverse()

    # Target counts per split
    total = len(records)
    targets = {
        "train": int(total * train_pct / 100),
        "eval": int(total * eval_pct / 100),
        "test": total - int(total * train_pct / 100) - int(total * eval_pct / 100),
    }

    assigned: dict[str, str] = {}  # rfp_id -> split name
    split_counts = {"train": 0, "eval": 0, "test": 0}

    # Phase 1: For each flag (rarest first), distribute its unassigned rfp_ids
    for flag in flags_by_rarity:
        # Collect unassigned rfp_ids that have this flag
        unassigned = [
            rid for rid, flags in rfp_flags.items() if flag in flags and rid not in assigned
        ]
        random.shuffle(unassigned)

        if not unassigned:
            continue

        # Distribute proportionally across splits
        n = len(unassigned)
        n_train = max(1, round(n * train_pct / 100))
        n_eval = max(1, round(n * eval_pct / 100))
        # Ensure we don't exceed total
        n_eval = min(n_eval, n - n_train)
        for rid in unassigned[:n_train]:
            assigned[rid] = "train"
            split_counts["train"] += len(by_rfp[rid])
        for rid in unassigned[n_train : n_train + n_eval]:
            assigned[rid] = "eval"
            split_counts["eval"] += len(by_rfp[rid])
        for rid in unassigned[n_train + n_eval :]:
            assigned[rid] = "test"
            split_counts["test"] += len(by_rfp[rid])

    # Phase 2: Assign remaining rfp_ids (no-flag or already-covered flags)
    unassigned_rest = [rid for rid in by_rfp if rid not in assigned]
    random.shuffle(unassigned_rest)

    for rid in unassigned_rest:
        # Assign to the split furthest below its target
        deficits = {s: targets[s] - split_counts[s] for s in targets}
        best = max(deficits, key=deficits.get)  # type: ignore[arg-type]
        assigned[rid] = best
        split_counts[best] += len(by_rfp[rid])

    # Build output lists
    train_recs: list[dict] = []
    eval_recs: list[dict] = []
    test_recs: list[dict] = []
    split_map = {"train": train_recs, "eval": eval_recs, "test": test_recs}

    for rfp_id, split_name in assigned.items():
        split_map[split_name].extend(by_rfp[rfp_id])

    return train_recs, eval_recs, test_recs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RAG seeds + train/eval/test from validated real data"
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "opus_validated_real.jsonl"),
        help="Input file (default: data/opus_validated_real.jsonl)",
    )
    parser.add_argument(
        "--split",
        default="80/10/10",
        help="Train/eval/test split percentages",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Max negatives as multiple of positives (default: 1.0)",
    )
    parser.add_argument(
        "--include-synthetic-seeds",
        action="store_true",
        help="Include synthetic seeds in ChromaDB (default: off)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without writing files",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Parse split
    parts = args.split.split("/")
    if len(parts) != 3:
        raise ValueError(f"Split must be train/eval/test like 80/10/10, got: {args.split}")
    train_pct, eval_pct, test_pct = (int(p) for p in parts)
    if train_pct + eval_pct + test_pct != 100:
        raise ValueError(f"Split must sum to 100, got {sum([train_pct, eval_pct, test_pct])}")

    # Step 0: Load and filter
    print("=== Step 0: Load data ===")
    input_path = Path(args.input)
    raw = load_jsonl(input_path)
    usable = [r for r in raw if r.get("validation") != "dropped"]
    print(f"  Loaded {len(raw)} total, {len(usable)} usable (non-dropped)")

    # Dedup by chunk_text
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in usable:
        if r["chunk_text"] not in seen:
            seen.add(r["chunk_text"])
            deduped.append(r)
    if len(deduped) < len(usable):
        print(f"  Removed {len(usable) - len(deduped)} duplicate chunks")
    usable = deduped

    flagged = sum(1 for r in usable if r.get("flags"))
    no_flag = sum(1 for r in usable if not r.get("flags"))
    print(f"  Flagged: {flagged} | No-flag: {no_flag}")

    # Step 1: RAG seeds
    print("\n=== Step 1: RAG seed selection ===")
    rag_seeds, remaining = select_rag_seeds(usable, RAG_PER_FLAG)
    seed_flags: Counter = Counter(s["flag"] for s in rag_seeds)
    print(f"  Reserved {len(rag_seeds)} RAG seeds:")
    for flag, count in sorted(seed_flags.items()):
        print(f"    {flag}: {count}")
    print(f"  Remaining: {len(remaining)}")

    # Step 2: Cap overrepresented flags
    print("\n=== Step 2: Cap overrepresented flags ===")
    remaining = cap_flags(remaining, FLAG_CAP)

    # Step 3: Cap negatives
    print("\n=== Step 3: Cap negatives ===")
    remaining = cap_negatives(remaining, ratio=args.negative_ratio)

    # Step 4: Stratified split by rfp_id
    print("\n=== Step 4: Stratified split by rfp_id ===")
    train, eval_set, test = split_by_rfp_id(remaining, train_pct, eval_pct, test_pct)
    unique_rfps = len({r["rfp_id"] for r in remaining})
    print(f"  {unique_rfps} unique rfp_ids")
    print(f"  Train: {len(train)} | Eval: {len(eval_set)} | Test: {len(test)}")

    # Per-flag breakdown per split
    for name, split in [("Train", train), ("Eval", eval_set), ("Test", test)]:
        flags: Counter = Counter()
        for r in split:
            for f in r.get("flags", []):
                flags[f] += 1
        neg = sum(1 for r in split if not r.get("flags"))
        print(f"\n  {name} flag distribution:")
        for f, c in flags.most_common():
            print(f"    {f}: {c}")
        print(f"    (no_flag): {neg}")

    if args.dry_run:
        print("\n(dry run — no files written)")
        return

    # Step 5: Build ChromaDB and wrap with RAG context
    print("\n=== Step 5: Build ChromaDB + wrap context ===")
    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()

    store = FlagRAGStore()

    # Add synthetic seeds
    synthetic_path = DATA_DIR / "synthetic" / "rag_seeds.jsonl"
    if args.include_synthetic_seeds and synthetic_path.exists():
        synthetic_seeds = load_jsonl(synthetic_path)
        added = store.add_passages(synthetic_seeds)
        print(f"  Added {added} synthetic seeds")

    # Add real seeds — prefix flag names to avoid ID collision with synthetic
    real_seeds_for_store = [
        {"flag": f"real_{s['flag']}", "passage": s["passage"]} for s in rag_seeds
    ]
    added = store.add_passages(real_seeds_for_store)
    print(f"  Added {added} real seeds")
    print(f"  Total in ChromaDB: {store.count()}")

    def wrap_split(split_records: list[dict]) -> list[dict]:
        wrapped = []
        for rec in split_records:
            chunk_text = rec["chunk_text"]
            flags = rec.get("flags", [])
            results = store.query(chunk_text, top_k=args.top_k + 1)
            results = [r for r in results if r["passage"] != chunk_text][: args.top_k]
            wrapped.append(make_training_record(system_prompt, chunk_text, flags, results))
        return wrapped

    train_wrapped = wrap_split(train)
    eval_wrapped = wrap_split(eval_set)
    test_wrapped = wrap_split(test)

    # Step 6: Write files
    print("\n=== Step 6: Write output ===")
    write_jsonl(DATA_DIR / "rag_exemplars.jsonl", rag_seeds)
    write_jsonl(DATA_DIR / "train.jsonl", train_wrapped)
    write_jsonl(DATA_DIR / "eval.jsonl", eval_wrapped)
    write_jsonl(DATA_DIR / "test.jsonl", test_wrapped)

    print(
        f"\nDone. Train: {len(train_wrapped)} | "
        f"Eval: {len(eval_wrapped)} | Test: {len(test_wrapped)} | "
        f"RAG seeds: {len(rag_seeds)}"
    )


if __name__ == "__main__":
    main()
