#!/usr/bin/env python3
"""Generate synthetic examples for rare flags and append to opus_validated_real.jsonl.

Generates realistic RFP chunks via Claude Opus (Bedrock) for flags with fewer
than a configurable threshold of examples. Doubles the count for each rare flag.

Usage:
  PYTHONPATH=. uv run python scripts/supplement_rare_flags.py
  PYTHONPATH=. uv run python scripts/supplement_rare_flags.py --threshold 40 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import anthropic

random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
DATA_FILE = REPO_ROOT / "data" / "opus_validated_real.jsonl"
FLAG_DEFS_PATH = REPO_ROOT / ".development-notes" / "notes" / "collated-flag-set.md"

MODEL = "us.anthropic.claude-opus-4-6"
BATCH_SIZE = 5

# Flag definitions for the 4 rare flags we know about, plus any others that
# fall under threshold. Keyed to the collated-flag-set descriptions.
FLAG_DEFINITIONS: dict[str, str] = {
    "hubzone_set_aside": (
        "Historically Underutilized Business Zone (HUBZone) set-aside. "
        "Look for FAR 52.219-13, HUBZone price evaluation preference, "
        "or explicit HUBZone set-aside language."
    ),
    "wosb_set_aside": (
        "Women-Owned Small Business (WOSB) set-aside. "
        "Look for FAR 52.219-29, WOSB Program set-aside, or explicit "
        "WOSB/women-owned language in the set-aside determination."
    ),
    "budget_too_low": (
        "Total contract budget is below $100K. The dollar figure should "
        "appear explicitly as total contract value, ceiling, or obligation amount."
    ),
    "8a_set_aside": (
        "8(a) Business Development Program set-aside. "
        "Look for FAR 52.219-11/14, 8(a) sole source or competitive set-aside, "
        "or explicit 8(a) program references."
    ),
    # Fallback definitions for any other flags that might fall under threshold
    "small_business_set_aside": "RFP is set aside exclusively for small businesses.",
    "sdvosb_set_aside": "Service-Disabled Veteran-Owned Small Business (SDVOSB) set-aside.",
    "large_team": "Scope requires 10 or more personnel.",
    "design_exercise": "Includes a design challenge, prototype submission, or hands-on exercise.",
    "marginal_short_duration": "Period of performance is less than 12 months.",
    "lpta_source_selection": (
        'Source selection is Lowest Price Technically Acceptable. Look for "LPTA."'
    ),
    "onsite_required": "All work must be performed at a specific location.",
    "brownfield": (
        "Contractor is taking over an existing codebase or continuing work from a prior team."
    ),
    "oral_presentation": ("Includes an oral presentation component as part of evaluation."),
    "agile_methodology": "Explicitly requires or expects Agile/Scrum methodology.",
    "off_the_shelf_software": (
        "Primary work is configuring or deploying commercial off-the-shelf platforms."
    ),
}


def load_records() -> list[dict]:
    with DATA_FILE.open() as f:
        return [json.loads(line) for line in f]


def count_flags(records: list[dict]) -> Counter:
    counts: Counter = Counter()
    for rec in records:
        if rec.get("validation") == "dropped":
            continue
        for flag in rec.get("flags", []):
            counts[flag] += 1
    return counts


def get_examples_for_flag(records: list[dict], flag: str, n: int = 3) -> list[str]:
    """Sample n random real chunks that have the given flag."""
    candidates = [
        rec["chunk_text"]
        for rec in records
        if rec.get("validation") != "dropped" and flag in rec.get("flags", [])
    ]
    return random.sample(candidates, min(n, len(candidates)))


def generate_prompt(flag: str, definition: str, n: int, examples: list[str]) -> str:
    examples_block = "\n\n---\n\n".join(f"Example {i + 1}:\n{ex}" for i, ex in enumerate(examples))
    return f"""Generate {n} realistic government RFP text chunks (200-400 words each)
that clearly and explicitly exhibit the following flag.

Flag: {flag}
Definition: {definition}

Here are real examples of RFP chunks that exhibit this flag. Match their tone,
specificity, and style — but do not copy them. Generate novel chunks for different
agencies and programs.

{examples_block}

Requirements:
- Write in authentic federal government solicitation language
- The flag must be clearly and explicitly present in each chunk
- Vary agency type, program context, and phrasing across chunks
- Each chunk should read as a natural excerpt from a larger RFP document
- Include realistic section numbers, FAR references, and procurement details
- Do not include flag labels or annotations in the text

Return a JSON array of {n} strings. No other text."""


def call_claude(client: anthropic.AnthropicBedrock, prompt: str) -> list[str]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.content[0].text.strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[1:end])
    return json.loads(content)


def make_record(flag: str, chunk_text: str, index: int) -> dict:
    return {
        "rfp_id": f"SYNTH-{flag}-{index:03d}",
        "chunk_index": 0,
        "chunk_text": chunk_text,
        "flags": [flag],
        "confidence": "synthetic",
        "adversarial": False,
        "validation": "keep",
        "original_flags": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplement rare flags with synthetic examples")
    parser.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="Flags below this count get supplemented (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without calling the API",
    )
    args = parser.parse_args()

    records = load_records()
    flag_counts = count_flags(records)

    # Find flags below threshold
    rare_flags = {flag: count for flag, count in flag_counts.items() if count < args.threshold}

    if not rare_flags:
        print(f"No flags below threshold ({args.threshold}). Nothing to do.")
        return

    print(f"Flags below threshold ({args.threshold}):")
    for flag, count in sorted(rare_flags.items(), key=lambda x: x[1]):
        to_generate = count  # double the count
        print(f"  {flag}: {count} → +{to_generate} = {count * 2}")

    total = sum(rare_flags.values())
    print(f"\nTotal to generate: {total}")

    if args.dry_run:
        print("(dry run — exiting)")
        return

    client = anthropic.AnthropicBedrock()
    new_records: list[dict] = []

    for flag, count in sorted(rare_flags.items(), key=lambda x: x[1]):
        to_generate = count
        definition = FLAG_DEFINITIONS.get(flag, flag)
        print(f"\n  Generating {to_generate} examples for {flag}...")

        generated: list[str] = []
        while len(generated) < to_generate:
            batch = min(BATCH_SIZE, to_generate - len(generated))
            examples = get_examples_for_flag(records, flag, n=3)
            chunks = call_claude(client, generate_prompt(flag, definition, batch, examples))
            generated.extend(chunks)
            print(f"    batch done ({len(generated)}/{to_generate})")

        for i, chunk_text in enumerate(generated[:to_generate]):
            new_records.append(make_record(flag, chunk_text, i))

    # Append to file
    with DATA_FILE.open("a") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nAppended {len(new_records)} synthetic records to {DATA_FILE.name}")

    # Show new totals
    all_records = records + new_records
    new_counts = count_flags(all_records)
    print("\nUpdated counts for supplemented flags:")
    for flag in sorted(rare_flags):
        print(f"  {flag}: {rare_flags[flag]} → {new_counts[flag]}")


if __name__ == "__main__":
    main()
