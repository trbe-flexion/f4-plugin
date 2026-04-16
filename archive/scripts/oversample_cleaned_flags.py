#!/usr/bin/env python3
"""Generate synthetic examples for cleaned flags (budget_too_low, large_team).

After clean_zero_recall_flags.py strips noisy examples, this script generates
new synthetic examples seeded from the remaining clean real examples.

Key differences from supplement_rare_flags.py:
  - Rotates 2-3 seed examples per API call (not always the same 3)
  - Includes explicit signal descriptions per flag
  - Tells Claude this trains a 3B model needing varied language + strong signal

Usage:
  PYTHONPATH=. uv run python scripts/oversample_cleaned_flags.py
  PYTHONPATH=. uv run python scripts/oversample_cleaned_flags.py --target 50 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anthropic

random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
DATA_FILE = REPO_ROOT / "data" / "opus_validated_real.jsonl"

MODEL = "us.anthropic.claude-opus-4-6-v1"
BATCH_SIZE = 5
SEEDS_PER_CALL = 3

FLAG_CONFIG: dict[str, dict] = {
    "budget_too_low": {
        "signal": (
            "The chunk contains an explicit total contract value, ceiling, or "
            "not-to-exceed (NTE) amount under $100K, stated as a dollar figure."
        ),
        "definition": (
            "Total contract budget is below $100K. The dollar figure must appear "
            "explicitly in the text as a total contract value, ceiling, obligation "
            "amount, or NTE — not as a line item, labor rate, or incidental amount."
        ),
    },
    "large_team": {
        "signal": (
            "The chunk explicitly states or enumerates 10 or more contractor "
            "personnel, FTEs, or named roles required for the effort."
        ),
        "definition": (
            "Scope requires 10+ people. The headcount or role enumeration must be "
            "explicit — either a stated number (e.g. '13 FTEs', '104 contractor "
            "personnel') or a list of 10+ named positions/roles."
        ),
    },
}


def load_records() -> list[dict]:
    with DATA_FILE.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def get_clean_seeds(records: list[dict], flag: str) -> list[dict]:
    """Get non-dropped, non-synthetic records that still have the flag."""
    return [
        rec
        for rec in records
        if rec.get("validation") != "dropped"
        and rec.get("confidence") != "synthetic"
        and flag in rec.get("flags", [])
    ]


def generate_prompt(flag: str, config: dict, n: int, seed_texts: list[str]) -> str:
    examples_block = "\n\n---\n\n".join(
        f"Example {i + 1}:\n{text}" for i, text in enumerate(seed_texts)
    )
    return f"""Generate {n} realistic government RFP text chunks (200-400 words each)
that clearly exhibit the following flag.

This data trains a 3B parameter model via fine-tuning. The model needs varied
language with a strong, consistent underlying signal.

Flag: {flag}
Definition: {config["definition"]}
Required signal: {config["signal"]}

Here are real examples of RFP chunks that exhibit this flag. Match their tone
and specificity, but generate novel chunks for different agencies, programs,
and contract types.

{examples_block}

Requirements:
- Write in authentic federal government solicitation language
- The required signal MUST be clearly present in every chunk
- Vary agency type, program context, section format, and phrasing across chunks
- Each chunk should read as a natural excerpt from a larger RFP document
- Include realistic section numbers, FAR references, and procurement details
- Do not include flag labels or annotations in the text
- Do not repeat or closely paraphrase the seed examples

Return a JSON array of {n} strings. No other text."""


def parse_json_array(text: str) -> list[str]:
    """Parse a JSON array from Claude's response, handling common issues."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Truncated response: try to salvage complete elements
    # Find the last complete string in the array
    results = []
    try:
        decoder = json.JSONDecoder()
        # Skip opening bracket
        idx = text.index("[") + 1
        while idx < len(text):
            idx = _skip_whitespace(text, idx)
            if idx >= len(text) or text[idx] == "]":
                break
            if text[idx] == ",":
                idx += 1
                continue
            try:
                obj, end_idx = decoder.raw_decode(text, idx)
                if isinstance(obj, str):
                    results.append(obj)
                idx = end_idx
            except json.JSONDecodeError:
                break
    except (ValueError, IndexError):
        pass

    if results:
        return results
    raise json.JSONDecodeError("Could not parse any elements", text, 0)


def _skip_whitespace(s: str, idx: int) -> int:
    while idx < len(s) and s[idx] in " \t\n\r":
        idx += 1
    return idx


MAX_RETRIES = 2


def call_claude(client: anthropic.AnthropicBedrock, prompt: str) -> list[str]:
    for attempt in range(MAX_RETRIES + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text
        try:
            return parse_json_array(content)
        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                print(f"    JSON parse failed, retrying ({attempt + 1}/{MAX_RETRIES})...")
            else:
                print(f"    JSON parse failed after {MAX_RETRIES + 1} attempts, skipping batch")
                return []


def make_record(flag: str, chunk_text: str, index: int) -> dict:
    return {
        "rfp_id": f"SYNTH-{flag}-v2-{index:03d}",
        "chunk_index": 0,
        "chunk_text": chunk_text,
        "flags": [flag],
        "confidence": "synthetic",
        "adversarial": False,
        "validation": "keep",
        "original_flags": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic examples for cleaned flags")
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Number of synthetic examples to generate per flag (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without calling the API",
    )
    args = parser.parse_args()

    records = load_records()

    for flag, config in FLAG_CONFIG.items():
        seeds = get_clean_seeds(records, flag)
        print(f"\n=== {flag} ===")
        print(f"  Clean seeds available: {len(seeds)}")
        print(f"  Target synthetic: {args.target}")

        if len(seeds) < 2:
            print("  SKIP: fewer than 2 clean seeds")
            continue

        if args.dry_run:
            print(f"  (dry run — would generate {args.target} examples)")
            continue

        client = anthropic.AnthropicBedrock()
        generated: list[str] = []

        while len(generated) < args.target:
            batch = min(BATCH_SIZE, args.target - len(generated))
            # Rotate seeds: random sample each call
            n_seeds = min(SEEDS_PER_CALL, len(seeds))
            sampled = random.sample(seeds, n_seeds)
            seed_texts = [s["chunk_text"] for s in sampled]

            chunks = call_claude(client, generate_prompt(flag, config, batch, seed_texts))
            generated.extend(chunks)
            print(f"    batch done ({len(generated)}/{args.target})")

        new_records = [
            make_record(flag, text, i) for i, text in enumerate(generated[: args.target])
        ]

        with DATA_FILE.open("a") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")

        print(f"  Appended {len(new_records)} records to {DATA_FILE.name}")

    if args.dry_run:
        print("\n(dry run — no files written)")


if __name__ == "__main__":
    main()
