#!/usr/bin/env python3
"""Label held-out test set RFPs using Claude Opus via Bedrock.

Reads RFP files (PDF/DOCX) or directories, chunks them into ~400-word pieces,
sends each chunk individually to Opus for flag labeling, and writes results
in training-ready messages format to data/test_reserve.jsonl.

Unlike label_real_data.py, this script:
  - Uses Opus (not Sonnet) for higher-quality ground truth
  - Chunks first, then labels each chunk individually
  - Has no per-flag quota tracking — labels everything found
  - Runs with async concurrency for speed
  - Writes directly to messages format

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
CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 50

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
        "The chunk describes procuring, configuring, or deploying a commercial "
        "off-the-shelf product as the PRIMARY work. Must contain explicit COTS "
        "language IN THE CHUNK BODY: 'off-the-shelf,' 'COTS,' 'GOTS,' 'NDI,' "
        "'FAR Part 12,' 'shrink-wrapped,' 'OOTB,' 'out-of-the-box.' A document "
        "title mentioning a platform name is NOT sufficient. Generic procurement "
        "clauses are NOT sufficient."
    ),
    "lpta_source_selection": (
        "Source selection is Lowest Price Technically Acceptable. "
        'Look for: "LPTA," "lowest price technically acceptable."'
    ),
    "small_business_set_aside": (
        "RFP is set aside exclusively for small businesses. Look for FAR 52.219-6, "
        '"small business set-aside," or explicit restriction.'
    ),
    "8a_set_aside": (
        "8(a) Business Development Program set-aside. Look for "
        'FAR 52.219-11/14, "8(a) sole source," "8(a) competitive set-aside."'
    ),
    "wosb_set_aside": (
        "Women-Owned Small Business (WOSB) or EDWOSB set-aside. "
        "Look for FAR 52.219-29/30, WOSB/EDWOSB language."
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
        "The chunk explicitly requires or describes Agile/Scrum methodology "
        "for the work. Look for: 'Agile,' 'Scrum,' 'sprint,' 'user stories,' "
        "'iterative development.' General mentions of modern practices or CI/CD "
        "alone are NOT sufficient — the chunk must specifically reference Agile "
        "as a methodology."
    ),
    "oral_presentation": (
        "The chunk describes an oral presentation as part of the evaluation/award "
        'process. Look for: "oral presentation," "oral proposal."'
    ),
    "design_exercise": (
        "The chunk describes a design challenge, prototype, proof of concept, or "
        "live demonstration required as part of evaluation. Look for: "
        "'demonstration,' 'challenge scenario,' 'proof of concept,' 'POC,' "
        "'flyoff,' 'pilot.'"
    ),
    "budget_too_low": (
        "The chunk states a total contract value, ceiling, or NTE under $100K "
        "as an explicit dollar figure."
    ),
    "onsite_required": (
        "The chunk explicitly requires all work at a specific location. Look for: "
        "'on-site,' 'onsite,' 'place of performance,' 'in-person,' 'physically "
        "present.' Do NOT flag if remote, hybrid, or telework options are mentioned."
    ),
    "large_team": (
        "The chunk explicitly states 10+ personnel/FTEs or enumerates 10+ named "
        "roles. Must be an explicit count or list, not inferred from scope size."
    ),
    "marginal_short_duration": (
        "The chunk explicitly states a period of performance under 12 months. "
        "Must be a stated duration, not inferred."
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


def chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping word-level chunks on paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if not para.strip():
            continue

        if current_words + para_words > chunk_words and current:
            chunks.append("\n\n".join(current))
            # Keep last paragraph(s) for overlap
            overlap: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                p_words = len(p.split())
                if overlap_count + p_words > overlap_words:
                    break
                overlap.insert(0, p)
                overlap_count += p_words
            current = overlap
            current_words = overlap_count

        current.append(para)
        current_words += para_words

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_prompt(chunk_text: str) -> str:
    flag_block = "\n".join(f"  - {flag}: {FLAG_DESCRIPTIONS[flag]}" for flag in KEEP_FLAGS)
    return f"""Your output will be parsed directly by a script. You MUST return ONLY
a JSON object in this exact format, with no other text:
{{"flags": ["flag_name_1", "flag_name_2"]}}
or if no flags: {{"flags": []}}

## Context

You are labeling government RFP text chunks for Flexion, a software consultancy
that screens RFPs to decide whether to bid. Each flag represents a specific
business signal that affects the bid/no-bid decision. These labels are ground
truth for evaluating a fine-tuned detection model — accuracy matters more than
coverage. A false positive is worse than a missed flag.

You are seeing one ~400-word chunk at a time, not the full document. You must
decide based ONLY on what is explicitly stated in this chunk. Do not infer flags
from context clues, document titles, boilerplate, or indirect language. Most
chunks contain no flags — that is expected and correct.

## Flags to detect:
{flag_block}

## Rules:
- Only flag what is EXPLICITLY stated in the chunk text
- When in doubt, return no flags — precision over recall
- Boilerplate, contract clauses, and legal terms are almost never flaggable
- Most chunks will have no flags. That is correct.

## Chunk:
{chunk_text}

Return ONLY: {{"flags": ["flag_name", ...]}} or {{"flags": []}}"""


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------


async def call_opus(
    client: anthropic.AsyncAnthropicBedrock,
    prompt: str,
    semaphore: asyncio.Semaphore,
    label: str,
) -> list[str]:
    """Call Opus to label a single chunk. Returns list of flag names."""
    max_retries = 3
    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=256,
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
                if isinstance(result, dict):
                    return result.get("flags", [])
                if isinstance(result, list):
                    return result
                print(f"  [warn] {label}: unexpected type {type(result).__name__}")
                return []

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
                if any(exc in item.name for exc in args.exclude):
                    continue
                entries.append(item)
        print(f"Found {len(entries)} RFP entries so far (added {source_dir.name})")

    if not entries:
        print("No RFP entries found.")
        return

    # Extract and chunk
    all_chunks: list[tuple[str, str, int]] = []  # (rfp_name, chunk_text, chunk_idx)
    for entry in entries:
        name = entry.stem
        text = extract_rfp_text(entry)
        if not text:
            print(f"  [skip] {name}: no text extracted")
            continue
        words = len(text.split())
        chunks = chunk_text(text)
        print(f"  {name}: {words} words, {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append((name, chunk, i))

    print(f"\nTotal chunks to label: {len(all_chunks)}")

    if args.dry_run:
        print("(dry run — no API calls)")
        return

    # Label concurrently
    client = anthropic.AsyncAnthropicBedrock()
    semaphore = asyncio.Semaphore(args.concurrency)
    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()

    # Resume support: load already-labeled chunks
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_texts: set[str] = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_texts.add(rec["messages"][1]["content"])
        if done_texts:
            print(f"  Resuming: {len(done_texts)} chunks already labeled")

    remaining = [(n, t, i) for n, t, i in all_chunks if t not in done_texts]
    print(f"  Chunks to label: {len(remaining)}")

    if not remaining:
        print("  All chunks already labeled.")
        # Still compute summary from existing file
        written = len(done_texts)
    else:
        from collections import Counter

        flag_counts: Counter = Counter()
        no_flag = 0
        written = len(done_texts)

        outfile = OUTPUT_FILE.open("a")

        async def label_chunk(rfp_name: str, text: str, idx: int) -> None:
            nonlocal written, no_flag
            label = f"{rfp_name}[{idx}]"
            prompt = build_prompt(text)
            flags = await call_opus(client, prompt, semaphore, label)
            # Filter to known flags
            flags = [f for f in flags if f in KEEP_FLAGS]
            assistant_content = "\n".join(flags) if flags else "no_flag"
            rec = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            outfile.write(json.dumps(rec) + "\n")
            outfile.flush()
            written += 1

            if assistant_content == "no_flag":
                no_flag += 1
            else:
                for flag in flags:
                    flag_counts[flag] += 1

        coros = [label_chunk(name, text, idx) for name, text, idx in remaining]
        await asyncio.gather(*coros)
        outfile.close()

    # Summary from full file
    from collections import Counter

    flag_counts_final: Counter = Counter()
    no_flag_final = 0
    total = 0
    with OUTPUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            content = rec["messages"][-1]["content"]
            if content == "no_flag":
                no_flag_final += 1
            else:
                for flag in content.strip().split("\n"):
                    flag_counts_final[flag] += 1

    flagged = total - no_flag_final
    print(f"\nTotal {total} chunks in {OUTPUT_FILE.name}")
    print(f"  Flagged: {flagged} | No-flag: {no_flag_final}")
    print("  Flag distribution:")
    for flag, count in flag_counts_final.most_common():
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
        "--exclude",
        nargs="*",
        default=[],
        help="Substrings to exclude from filenames (e.g. 'Sources Sought' 'Capability')",
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
