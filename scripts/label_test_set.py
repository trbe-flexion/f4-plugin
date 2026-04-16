#!/usr/bin/env python3
"""Label held-out test set RFPs using Claude Opus via Bedrock.

Reads individual RFP files (PDF/DOCX) or directories, chunks them, sends
each section to Opus for flag extraction, and writes labeled chunks to
data/test_labeled.jsonl.

Unlike label_real_data.py, this script:
  - Uses Opus (not Sonnet) for higher-quality ground truth
  - Has no per-flag quota tracking — extracts everything found
  - Runs with async concurrency for speed
  - Uses the current (cleaned) flag set

Usage:
  PYTHONPATH=. uv run python scripts/label_test_set.py "/path/to/Pursue" "/path/to/Do Not Pursue"
  PYTHONPATH=. uv run python scripts/label_test_set.py "/path/to/Pursue" --concurrency 5 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import anthropic

from src.frontend.extraction import extract_text

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "test_reserve.jsonl"
SYSTEM_PROMPT_PATH = REPO_ROOT / "scripts" / "system-prompt.md"

MODEL = "us.anthropic.claude-opus-4-6-v1"
MAX_SECTION_WORDS = 30_000

# Current flag set (post-cleaning, matches system-prompt.md)
KEEP_FLAGS: list[str] = [
    "off_the_shelf_software",
    "lpta_source_selection",
    "small_business_set_aside",
    "8a_set_aside",
    "wosb_set_aside",
    "sdvosb_set_aside",
    "hubzone_set_aside",
    "agile_methodology",
    "oral_presentation",
    "design_exercise",
    "budget_too_low",
    "onsite_required",
    "large_team",
    "marginal_short_duration",
]

FLAG_DESCRIPTIONS: dict[str, str] = {
    "off_the_shelf_software": (
        "Primary work is configuring/deploying COTS platforms. Look for explicit COTS "
        "language: 'off-the-shelf,' 'COTS,' 'GOTS,' 'NDI,' 'FAR Part 12,' "
        "'shrink-wrapped,' 'OOTB / out-of-the-box.' Do NOT flag incidental mentions "
        "of commercial software tools within a custom development RFP."
    ),
    "lpta_source_selection": (
        "Source selection is Lowest Price Technically Acceptable. "
        'Look for: "LPTA," "lowest price technically acceptable."'
    ),
    "small_business_set_aside": (
        "RFP is set aside exclusively for small businesses. Look for FAR 52.219-6, "
        '"small business set-aside," or explicit small business restriction.'
    ),
    "8a_set_aside": (
        "8(a) Business Development Program set-aside. Look for FAR 52.219-11/14, "
        '"8(a) sole source," or "8(a) competitive set-aside."'
    ),
    "wosb_set_aside": (
        "Women-Owned Small Business (WOSB) or Economically Disadvantaged WOSB "
        "(EDWOSB) set-aside. Look for FAR 52.219-29/30, WOSB/EDWOSB language."
    ),
    "sdvosb_set_aside": (
        "Service-Disabled Veteran-Owned Small Business set-aside. "
        "Look for FAR 52.219-27, SDVOSB language."
    ),
    "hubzone_set_aside": (
        "HUBZone set-aside. Look for FAR 52.219-13, HUBZone price evaluation "
        "preference, or explicit HUBZone set-aside language."
    ),
    "agile_methodology": (
        "RFP explicitly requires or expects Agile/Scrum methodology. "
        'Look for: "Agile," "Scrum," "sprint," "user stories," "iterative."'
    ),
    "oral_presentation": (
        "RFP includes an oral presentation component as part of evaluation. "
        'Look for: "oral presentation," "oral proposal," "oral evaluation."'
    ),
    "design_exercise": (
        "RFP includes a design challenge, prototype, proof of concept, or "
        "demonstration as part of evaluation. Look for: 'demonstration,' "
        "'challenge scenario,' 'proof of concept,' 'POC,' 'flyoff,' 'pilot.'"
    ),
    "budget_too_low": (
        "Total contract budget is below $100K. The dollar figure must appear "
        "explicitly as a total contract value, ceiling, obligation amount, or NTE."
    ),
    "onsite_required": (
        "All work must be performed at a specific location. Look for explicit "
        "onsite language: 'on-site,' 'onsite,' 'place of performance,' "
        "'in-person,' 'physically present.' Do NOT flag if hybrid, remote, "
        "or flexible options are offered."
    ),
    "large_team": (
        "Scope requires 10+ people. Must be explicit — a stated number "
        "(e.g. '13 FTEs,' '104 contractor personnel') or a list of 10+ "
        "named positions/roles."
    ),
    "marginal_short_duration": (
        "Period of performance is less than 12 months. Must be explicit — "
        "a stated duration, not inferred from context."
    ),
}


# ---------------------------------------------------------------------------
# Text extraction and chunking
# ---------------------------------------------------------------------------


def extract_file_text(path: Path) -> str:
    """Extract text from a single PDF or DOCX file."""
    try:
        text = extract_text(str(path))
        return text.strip()
    except Exception as e:
        print(f"  [warn] Failed to extract {path.name}: {e}")
        return ""


def extract_rfp_text(path: Path) -> str:
    """Extract text from a file or all PDF/DOCX files in a directory."""
    supported = {".pdf", ".docx"}
    if path.is_file() and path.suffix.lower() in supported:
        return extract_file_text(path)

    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() in supported)
        parts = [extract_file_text(f) for f in files]
        return "\n\n".join(p for p in parts if p)

    return ""


def split_into_sections(text: str, max_words: int = MAX_SECTION_WORDS) -> list[str]:
    """Split text into sections of roughly max_words on paragraph boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    paragraphs = text.split("\n\n")
    sections: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_word_list = para.split()
        para_words = len(para_word_list)
        if para_words > max_words:
            if current:
                sections.append("\n\n".join(current))
                current = []
                current_words = 0
            for j in range(0, para_words, max_words):
                sections.append(" ".join(para_word_list[j : j + max_words]))
            continue
        if current_words + para_words > max_words and current:
            sections.append("\n\n".join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words

    if current:
        sections.append("\n\n".join(current))
    return sections


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_prompt(rfp_text: str) -> str:
    flag_block = "\n".join(f"  - {flag}: {FLAG_DESCRIPTIONS[flag]}" for flag in KEEP_FLAGS)
    return f"""You are labeling government RFP text to build a ground-truth evaluation
dataset for a fine-tuned 3B flag detection model. Accuracy is critical — these labels
will be used to measure model performance.

Read the RFP text below and extract passages (~200-400 words each) that exhibit
any of the flags listed. Also extract 1-2 passages that clearly contain NO flags
(for no_flag evaluation data).

## Flags to detect:
{flag_block}

## Instructions:
- Each passage should be a self-contained chunk with clear, explicit evidence
- Only label a flag if the signal is unambiguous in the chunk text itself
- For each passage, report:
  - chunk_text: the extracted passage (200-400 words)
  - flags: list of flag names present (use exact names above), or empty list for no_flag
  - confidence: "high" or "medium" (only include "high" confidence labels)
- Skip boilerplate (tables of contents, standard clauses) unless they contain flag signals
- If no flags are found, return only no_flag passages

Return a JSON array. No other text.

Example:
```json
[
  {{
    "chunk_text": "This solicitation is set-aside for SDVOSB businesses...",
    "flags": ["sdvosb_set_aside"],
    "confidence": "high"
  }},
  {{
    "chunk_text": "The contractor shall provide IT modernization services...",
    "flags": [],
    "confidence": "high"
  }}
]
```

## RFP Text:
{rfp_text}"""


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------


async def call_opus(
    client: anthropic.AsyncAnthropicBedrock,
    prompt: str,
    semaphore: asyncio.Semaphore,
    label: str,
) -> list[dict]:
    """Call Opus with retry on throttling."""
    max_retries = 3
    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                )
                if not response.content:
                    print(f"  [warn] {label}: empty response")
                    return []

                content = response.content[0].text.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    end = -1 if lines[-1].strip() == "```" else len(lines)
                    content = "\n".join(lines[1:end])

                result = json.loads(content)
                if not isinstance(result, list):
                    print(f"  [warn] {label}: expected array, got {type(result).__name__}")
                    return []
                return result

            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                print(f"  [throttle] {label}: retry in {wait}s")
                await asyncio.sleep(wait)
            except json.JSONDecodeError as e:
                print(f"  [warn] {label}: JSON parse error: {e}")
                return []
            except Exception as e:
                print(f"  [error] {label}: {e}")
                return []
    print(f"  [fail] {label}: exhausted retries")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def process_rfps(args: argparse.Namespace) -> None:
    # Collect all RFP files/dirs from all source directories
    supported = {".pdf", ".docx"}
    skip_names = {".DS_Store", "desktop.ini"}
    entries: list[Path] = []

    for source in args.source_dirs:
        source_dir = Path(source)
        if not source_dir.exists():
            print(f"Source directory not found: {source_dir}")
            continue
        for item in sorted(source_dir.iterdir()):
            if item.name in skip_names or item.name.startswith("."):
                continue
            if item.suffix == ".zip":
                continue
            if (item.is_file() and item.suffix.lower() in supported) or item.is_dir():
                entries.append(item)
        print(f"Found {len(entries)} RFP entries so far (added {source_dir.name})")

    if not entries:
        print("No RFP entries found.")
        return

    # Extract and chunk
    all_tasks: list[tuple[str, str, int]] = []  # (rfp_name, section_text, section_idx)
    for entry in entries:
        name = entry.stem
        text = extract_rfp_text(entry)
        if not text:
            print(f"  [skip] {name}: no text extracted")
            continue
        words = len(text.split())
        sections = split_into_sections(text)
        print(f"  {name}: {words} words, {len(sections)} section(s)")
        for i, section in enumerate(sections):
            all_tasks.append((name, section, i))

    print(f"\nTotal sections to label: {len(all_tasks)}")

    if args.dry_run:
        print("(dry run — no API calls)")
        return

    # Label concurrently
    client = anthropic.AsyncAnthropicBedrock()
    semaphore = asyncio.Semaphore(args.concurrency)

    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()

    async def label_section(rfp_name: str, section: str, idx: int) -> list[dict]:
        label = f"{rfp_name}[{idx}]"
        prompt = build_prompt(section)
        chunks = await call_opus(client, prompt, semaphore, label)
        print(f"  {label}: {len(chunks)} chunks extracted")
        results = []
        for chunk in chunks:
            if not isinstance(chunk, dict) or "chunk_text" not in chunk:
                continue
            flags = chunk.get("flags", [])
            assistant_content = "\n".join(flags) if flags else "no_flag"
            results.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk["chunk_text"]},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )
        return results

    coros = [label_section(name, section, idx) for name, section, idx in all_tasks]
    all_results = await asyncio.gather(*coros)

    # Flatten and write
    records = [rec for batch in all_results for rec in batch]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Summary
    from collections import Counter

    flag_counts: Counter = Counter()
    no_flag = 0
    for rec in records:
        label = rec["messages"][-1]["content"]
        if label == "no_flag":
            no_flag += 1
        else:
            for flag in label.strip().split("\n"):
                flag_counts[flag] += 1

    print(f"\nWrote {len(records)} chunks to {OUTPUT_FILE.name}")
    flagged = len(records) - no_flag
    print(f"  Flagged: {flagged} | No-flag: {no_flag}")
    print("  Flag distribution:")
    for flag, count in flag_counts.most_common():
        print(f"    {flag}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label test set RFPs with Opus")
    parser.add_argument(
        "source_dirs",
        nargs="+",
        help="One or more directories containing RFP files/folders to label",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent Opus calls (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show extraction plan without calling API",
    )
    args = parser.parse_args()
    asyncio.run(process_rfps(args))


if __name__ == "__main__":
    main()
