"""Validation pass: Opus reviews Sonnet's labels in labeled_real.jsonl.

Sends batches of chunks to Claude Opus for review. Each chunk gets a verdict:
KEEP (labels correct), FIX (corrected flags), or DROP (too noisy for training).

Usage:
  PYTHONPATH=. uv run python scripts/validate_labels.py
  PYTHONPATH=. uv run python scripts/validate_labels.py --batch-size 10
  PYTHONPATH=. uv run python scripts/validate_labels.py --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import anthropic

MODEL = "us.anthropic.claude-opus-4-0-20250514"
MAX_CONCURRENCY = 5
DEFAULT_BATCH_SIZE = 10

KEEP_FLAGS: list[str] = [
    "waterfall_methodology",
    "off_the_shelf_software",
    "no_custom_development",
    "lpta_source_selection",
    "small_business_set_aside",
    "8a_set_aside",
    "wosb_set_aside",
    "edwosb_set_aside",
    "sdvosb_set_aside",
    "hubzone_set_aside",
    "agile_methodology",
    "oral_presentation",
    "design_exercise",
    "budget_too_low",
    "brownfield",
    "onsite_required",
    "onsite_madison",
    "large_team",
    "marginal_short_duration",
]

FLAG_DESCRIPTIONS: dict[str, str] = {
    "waterfall_methodology": (
        "Explicitly requires sequential/waterfall SDLC. Must mandate waterfall as the "
        "development methodology — project phases, contract phases, and hardware installation "
        "sequences do NOT count."
    ),
    "off_the_shelf_software": (
        "Primary work is configuring/deploying COTS platforms (Salesforce, SharePoint, etc.) "
        'or "off-the-shelf," "commercial solution," "configure and deploy."'
    ),
    "no_custom_development": (
        "Explicitly excludes custom software development. The RFP must say no custom "
        "development is wanted — purchasing a product or subscription does NOT qualify."
    ),
    "lpta_source_selection": (
        'Source selection is Lowest Price Technically Acceptable. "LPTA," '
        '"lowest price technically acceptable."'
    ),
    "small_business_set_aside": "RFP is set aside exclusively for small businesses.",
    "8a_set_aside": "8(a) Business Development Program set-aside.",
    "wosb_set_aside": "Women-Owned Small Business (WOSB) set-aside.",
    "edwosb_set_aside": "Economically Disadvantaged WOSB (EDWOSB) set-aside.",
    "sdvosb_set_aside": "Service-Disabled Veteran-Owned Small Business (SDVOSB) set-aside.",
    "hubzone_set_aside": "HUBZone set-aside.",
    "agile_methodology": "Explicitly requires or expects Agile/Scrum methodology.",
    "oral_presentation": "Includes an oral presentation component in evaluation.",
    "design_exercise": "Includes a design challenge, prototype, or exercise.",
    "budget_too_low": "Total contract budget is below $100K.",
    "brownfield": (
        "Taking over an existing codebase or continuing from a prior development team. "
        "System replacement (e.g., buying new hardware to replace old) does NOT count."
    ),
    "onsite_required": (
        "All work must be performed at a specific location (not Madison, WI). "
        "Hybrid/remote/flexible options mean this flag does NOT apply."
    ),
    "onsite_madison": "Onsite work required and the location is Madison, WI.",
    "large_team": (
        "Scope requires 10 or more personnel on the contractor team. "
        "User counts, vendor counts, and site counts are NOT team size."
    ),
    "marginal_short_duration": (
        "Period of performance is strictly less than 12 months. 12 months exactly does NOT qualify."
    ),
}

VALID_FLAGS = set(KEEP_FLAGS)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_validation_prompt(batch: list[dict]) -> str:
    """Build the Opus validation prompt for a batch of chunks."""
    flag_block = "\n".join(f"  - {flag}: {FLAG_DESCRIPTIONS[flag]}" for flag in KEEP_FLAGS)

    chunks_block = ""
    for idx, chunk in enumerate(batch):
        flags_str = ", ".join(chunk["flags"]) if chunk["flags"] else "(no flags)"
        chunks_block += f"""
--- Chunk {idx} ---
Flags labeled by Sonnet: {flags_str}
Confidence: {chunk.get("confidence", "medium")}
Adversarial: {chunk.get("adversarial", False)}

Text:
{chunk["chunk_text"]}
"""

    return f"""You are reviewing flag labels assigned by Claude Sonnet to real government RFP text.
Your job is to audit each chunk and verdict it for use as training data for a small
flag-detection model.

## Flag Definitions
{flag_block}

## Common Sonnet Errors to Watch For
- no_custom_development: Sonnet labels product purchases (CrowdStrike, Pluralsight licenses)
  as "no custom development." Buying a product ≠ explicitly excluding custom dev.
- large_team: Sonnet triggers on user counts, vendor counts, or site counts instead of
  actual contractor team size (10+ personnel).
- marginal_short_duration: Sonnet labels 12-month periods as "less than 12 months."
  12 months exactly does NOT qualify.
- waterfall_methodology: Sonnet triggers on project phases, contract phases, and hardware
  installation sequences. Only actual waterfall SDLC mandates count.
- brownfield: Sonnet labels system replacement (buying new to replace old) as brownfield.
  Only taking over an existing codebase counts.

## Instructions
For each chunk, output one verdict:
- KEEP: All labels are correct. No changes needed.
- FIX: One or more labels are wrong. Provide the corrected flag list.
  You may remove incorrect flags AND add missing flags, but only add a flag if
  the evidence is unambiguous.
- DROP: The chunk is too noisy, ambiguous, or low-quality to be useful training data.
  Borderline examples hurt LoRA fine-tuning more than they help.

Return a JSON array with one object per chunk, in the same order:
```json
[
  {{"index": 0, "verdict": "KEEP"}},
  {{"index": 1, "verdict": "FIX", "corrected_flags": ["flag_a"]}},
  {{"index": 2, "verdict": "DROP", "reason": "ambiguous scope"}}
]
```

For KEEP: only index and verdict needed.
For FIX: include corrected_flags (the full corrected list, not just changes).
For DROP: include a brief reason.

Return ONLY the JSON array.

## Chunks to Review
{chunks_block}"""


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------


async def call_opus(
    client: anthropic.AsyncAnthropicBedrock,
    prompt: str,
    semaphore: asyncio.Semaphore,
    batch_idx: int,
) -> list[dict]:
    """Call Opus with retry on throttling."""
    max_retries = 3
    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                if not response.content:
                    print(f"  [warn] Batch {batch_idx}: empty response")
                    return []

                content = response.content[0].text.strip()

                # Strip markdown code fences
                if content.startswith("```"):
                    lines = content.split("\n")
                    end = -1 if lines[-1].strip() == "```" else len(lines)
                    content = "\n".join(lines[1:end])

                result = json.loads(content)
                if not isinstance(result, list):
                    print(
                        f"  [warn] Batch {batch_idx}: expected array, got {type(result).__name__}"
                    )
                    return []
                return result

            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                print(f"  [throttle] Batch {batch_idx}: retry in {wait}s")
                await asyncio.sleep(wait)
            except anthropic.BadRequestError as e:
                print(f"  [warn] Batch {batch_idx}: API rejected: {e}")
                return []
            except json.JSONDecodeError as e:
                print(f"  [warn] Batch {batch_idx}: JSON parse failed: {e}")
                return []
            except Exception as e:
                print(f"  [warn] Batch {batch_idx}: unexpected error: {e}")
                return []

    print(f"  [error] Batch {batch_idx}: exhausted retries")
    return []


# ---------------------------------------------------------------------------
# Apply verdicts
# ---------------------------------------------------------------------------


def apply_verdicts(batch: list[dict], verdicts: list[dict]) -> list[dict]:
    """Apply Opus verdicts to the original chunks."""
    results = []
    verdict_map = {v["index"]: v for v in verdicts if "index" in v}

    for idx, chunk in enumerate(batch):
        verdict = verdict_map.get(idx)
        if verdict is None:
            # Missing verdict — keep original as safety fallback
            chunk["validation"] = "keep"
            chunk["original_flags"] = chunk["flags"]
            results.append(chunk)
            continue

        v = verdict.get("verdict", "").upper()
        original_flags = list(chunk["flags"])

        if v == "KEEP":
            chunk["validation"] = "keep"
            chunk["original_flags"] = original_flags
            results.append(chunk)
        elif v == "FIX":
            corrected = verdict.get("corrected_flags", [])
            # Validate corrected flags
            valid_corrected = [f for f in corrected if f in VALID_FLAGS]
            chunk["original_flags"] = original_flags
            chunk["flags"] = valid_corrected
            chunk["validation"] = "fixed"
            results.append(chunk)
        elif v == "DROP":
            chunk["validation"] = "dropped"
            chunk["original_flags"] = original_flags
            chunk["drop_reason"] = verdict.get("reason", "")
            results.append(chunk)
        else:
            # Unknown verdict — keep as fallback
            chunk["validation"] = "keep"
            chunk["original_flags"] = original_flags
            results.append(chunk)

    return results


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def print_stats(all_results: list[dict]) -> None:
    """Print validation summary."""
    kept = sum(1 for r in all_results if r["validation"] == "keep")
    fixed = sum(1 for r in all_results if r["validation"] == "fixed")
    dropped = sum(1 for r in all_results if r["validation"] == "dropped")
    total = len(all_results)

    print(f"\n{'=' * 60}")
    print(f"Validation complete: {total} chunks")
    print(f"  KEEP:  {kept} ({100 * kept / total:.1f}%)")
    print(f"  FIX:   {fixed} ({100 * fixed / total:.1f}%)")
    print(f"  DROP:  {dropped} ({100 * dropped / total:.1f}%)")

    # Per-flag changes
    flags_added: dict[str, int] = {}
    flags_removed: dict[str, int] = {}
    for r in all_results:
        if r["validation"] != "fixed":
            continue
        orig = set(r.get("original_flags", []))
        curr = set(r["flags"])
        for f in curr - orig:
            flags_added[f] = flags_added.get(f, 0) + 1
        for f in orig - curr:
            flags_removed[f] = flags_removed.get(f, 0) + 1

    if flags_added or flags_removed:
        print("\n  Flag changes (FIX verdicts):")
        all_flags = sorted(set(flags_added) | set(flags_removed))
        for f in all_flags:
            added = flags_added.get(f, 0)
            removed = flags_removed.get(f, 0)
            parts = []
            if added:
                parts.append(f"+{added}")
            if removed:
                parts.append(f"-{removed}")
            print(f"    {f}: {', '.join(parts)}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    """Main async entry point."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load input
    chunks = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {input_path}")

    # Resume support
    processed_count = 0
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    processed_count += 1
        print(f"Resuming: {processed_count} chunks already validated")
        chunks = chunks[processed_count:]

    if args.limit:
        chunks = chunks[: args.limit]
        print(f"Limiting to {args.limit} chunks")

    if not chunks:
        print("Nothing to validate.")
        return

    # Batch
    batches = []
    for i in range(0, len(chunks), args.batch_size):
        batches.append(chunks[i : i + args.batch_size])
    print(
        f"Processing {len(chunks)} chunks in {len(batches)} batches "
        f"(batch size {args.batch_size}, concurrency {MAX_CONCURRENCY})"
    )

    client = anthropic.AsyncAnthropicBedrock()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    all_results: list[dict] = []
    # Load existing results for stats if resuming
    if args.resume and processed_count > 0:
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    mode = "a" if args.resume and processed_count > 0 else "w"
    start_time = time.time()
    api_calls = 0

    with open(output_path, mode) as out_f:
        # Process batches in groups of MAX_CONCURRENCY
        for group_start in range(0, len(batches), MAX_CONCURRENCY):
            group = batches[group_start : group_start + MAX_CONCURRENCY]
            tasks = []
            for bi, batch in enumerate(group):
                batch_idx = group_start + bi
                prompt = build_validation_prompt(batch)
                tasks.append(call_opus(client, prompt, semaphore, batch_idx))

            results = await asyncio.gather(*tasks)
            api_calls += len(group)

            for bi, (batch, verdicts) in enumerate(zip(group, results, strict=True)):
                batch_idx = group_start + bi
                if verdicts:
                    validated = apply_verdicts(batch, verdicts)
                else:
                    # API failure — keep originals as fallback
                    validated = []
                    for chunk in batch:
                        chunk["validation"] = "keep"
                        chunk["original_flags"] = list(chunk["flags"])
                        validated.append(chunk)

                for record in validated:
                    out_f.write(json.dumps(record) + "\n")
                all_results.extend(validated)
            out_f.flush()

            # Progress
            chunks_done = (
                min((group_start + len(group)) * args.batch_size, len(chunks)) + processed_count
            )
            total_chunks = len(chunks) + processed_count
            elapsed = time.time() - start_time
            rate = (chunks_done - processed_count) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{chunks_done}/{total_chunks}] {api_calls} API calls | {rate:.0f} chunks/min")

    print_stats(all_results)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"API calls: {api_calls}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Sonnet labels with Opus")
    parser.add_argument(
        "--input",
        default="data/labeled_real.jsonl",
        help="Input labeled file (default: data/labeled_real.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/validated_real.jsonl",
        help="Output validated file (default: data/validated_real.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Chunks per API call (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N chunks then stop (for test runs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
