#!/usr/bin/env python3
"""Label real RFP data using Claude via Bedrock.

Reads RFP directories (PDF/DOCX), sends text to Claude for flag extraction
with per-flag quota tracking, outputs data/labeled_real.jsonl.

Usage:
  PYTHONPATH=. uv run python scripts/label_real_data.py \
      --source-dir "/path/to/downloads" --target-per-flag 10 --dry-run
  PYTHONPATH=. uv run python scripts/label_real_data.py \
      --source-dir "/path/to/downloads" --target-per-flag 150 --seed 42

Requires AWS credentials in environment with bedrock:InvokeModel permission.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import anthropic

from src.frontend.extraction import extract_text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
FLAG_DEFS_PATH = REPO_ROOT / ".development-notes" / "collated-flag-set.md"

MODEL = "us.anthropic.claude-sonnet-4-6"
MAX_SECTION_WORDS = 30_000

# The 19 flags the small model detects (from collated-flag-set.md "Keep" section)
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
        'Explicitly requires sequential/waterfall development. Look for: "waterfall," '
        '"traditional SDLC," "sequential phases," "fixed requirements baseline," '
        '"phase gate," "V-model."'
    ),
    "off_the_shelf_software": (
        "Primary work is configuring/deploying COTS platforms. Look for product names "
        '(Salesforce, SharePoint, WordPress) or "off-the-shelf," "commercial solution," '
        '"configure and deploy."'
    ),
    "no_custom_development": (
        'Explicitly excludes custom software development. "no custom development," '
        '"configuration only," "COTS solution required."'
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
        "Taking over existing codebase or continuing from prior team. "
        '"existing system," "transition from incumbent," "legacy codebase."'
    ),
    "onsite_required": (
        "All work must be performed at a specific location (not Madison, WI). "
        "Not if hybrid/remote/flexible options offered."
    ),
    "onsite_madison": "Onsite work required and the location is Madison, WI.",
    "large_team": "Scope requires 10 or more personnel.",
    "marginal_short_duration": "Period of performance is less than 12 months.",
}


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_rfp_text(rfp_dir: Path) -> str:
    """Extract and combine text from all PDF/DOCX files in an RFP directory."""
    supported = {".pdf", ".docx"}
    files = sorted(f for f in rfp_dir.iterdir() if f.is_file() and f.suffix.lower() in supported)
    if not files:
        return ""

    parts: list[str] = []
    for f in files:
        try:
            text = extract_text(str(f))
            if text.strip():
                parts.append(text)
        except Exception as e:
            print(f"    [warn] Failed to extract {f.name}: {e}")
    return "\n\n".join(parts)


def split_into_sections(text: str, max_words: int = MAX_SECTION_WORDS) -> list[str]:
    """Split text into sections of roughly max_words, breaking on paragraph boundaries."""
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
        # Hard-split oversized paragraphs (e.g., PDFs with no line breaks)
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
# Claude extraction
# ---------------------------------------------------------------------------


def build_extraction_prompt(
    rfp_text: str,
    needed_flags: list[str],
    filled_flags: list[str],
) -> str:
    """Build the prompt for Claude to extract flag-relevant passages."""
    flag_block = "\n".join(f"  - {flag}: {FLAG_DESCRIPTIONS[flag]}" for flag in needed_flags)
    filled_block = ""
    if filled_flags:
        filled_block = (
            "\n\nThe following flags have reached their quota — do NOT extract "
            "passages for them:\n" + ", ".join(filled_flags)
        )

    return f"""You are labeling real government RFP text for a flag detection training dataset.

Read the RFP text below and extract passages (~200-400 words each) that exhibit
any of the flags listed. Also extract 1-2 passages that clearly contain NO flags
(for no-flag training data).

## Flags to look for:
{flag_block}
{filled_block}

## Instructions:
- Each extracted passage should be a self-contained chunk with clear evidence for the flag(s)
- Choose natural chunk boundaries — the passage should make sense on its own
- For each passage, report:
  - chunk_text: the extracted passage (200-400 words)
  - flags: list of flag names present (use exact names above), or empty list for no-flag
  - confidence: "high" or "medium" (how clearly the flag is present)
  - adversarial: true if the passage mentions flag-related concepts but the flag does NOT apply
    (e.g., mentions "waterfall" but says they won't use it). These are valuable training negatives.
- Skip boilerplate sections (tables of contents, standard clauses) unless they contain flag signals
- If no flags from the needed list are found, return only no-flag passages or an empty array

Return a JSON array. No other text.

Example:
```json
[
  {{
    "chunk_text": "This solicitation is set-aside for SDVOSB businesses...",
    "flags": ["sdvosb_set_aside"],
    "confidence": "high",
    "adversarial": false
  }},
  {{
    "chunk_text": "The contractor shall provide IT modernization services...",
    "flags": [],
    "confidence": "high",
    "adversarial": false
  }}
]
```

## RFP Text:
{rfp_text}"""


def call_claude(
    client: anthropic.AnthropicBedrock,
    prompt: str,
) -> list[dict]:
    """Call Claude and parse the JSON array response."""
    # Guard: skip prompts that are likely to exceed token limits
    # (~0.75 tokens per word, plus prompt overhead)
    prompt_words = len(prompt.split())
    if prompt_words > 600_000:
        print(f"    [warn] Prompt too large ({prompt_words} words), skipping")
        return []

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.BadRequestError as e:
        print(f"    [warn] API rejected request: {e}")
        return []

    if not response.content:
        print("    [warn] Claude returned empty content, skipping")
        return []
    content = response.content[0].text.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[1:end])

    try:
        result = json.loads(content)
        if not isinstance(result, list):
            print(f"    [warn] Expected JSON array, got {type(result).__name__}")
            return []
        return result
    except json.JSONDecodeError as e:
        print(f"    [warn] JSON parse failed: {e}")
        print(f"    [warn] Response tail: ...{content[-200:]}")
        return []


def validate_chunk(chunk: dict) -> bool:
    """Validate a chunk dict has required fields with correct types."""
    if not isinstance(chunk, dict):
        return False
    if "chunk_text" not in chunk or "flags" not in chunk:
        return False
    if not isinstance(chunk["flags"], list):
        return False
    # Validate flag names
    valid_flags = set(KEEP_FLAGS)
    return all(flag in valid_flags for flag in chunk["flags"])


# ---------------------------------------------------------------------------
# Quota tracking
# ---------------------------------------------------------------------------


class QuotaTracker:
    """Track per-flag extraction quotas."""

    def __init__(self, target_per_flag: int, target_no_flag: int):
        self.target_per_flag = target_per_flag
        self.target_no_flag = target_no_flag
        self.counts: dict[str, int] = {flag: 0 for flag in KEEP_FLAGS}
        self.no_flag_count: int = 0

    def update(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            flags = chunk.get("flags", [])
            if not flags:
                self.no_flag_count += 1
            else:
                for flag in flags:
                    if flag in self.counts:
                        self.counts[flag] += 1

    def needed_flags(self) -> list[str]:
        return [f for f in KEEP_FLAGS if self.counts[f] < self.target_per_flag]

    def filled_flags(self) -> list[str]:
        return [f for f in KEEP_FLAGS if self.counts[f] >= self.target_per_flag]

    def all_filled(self) -> bool:
        return (
            all(c >= self.target_per_flag for c in self.counts.values())
            and self.no_flag_count >= self.target_no_flag
        )

    def total(self) -> int:
        return sum(self.counts.values()) + self.no_flag_count

    def status_line(self) -> str:
        filled = len(self.filled_flags())
        total_flags = len(KEEP_FLAGS)
        return (
            f"Total: {self.total()} chunks | "
            f"Flags filled: {filled}/{total_flags} | "
            f"No-flag: {self.no_flag_count}/{self.target_no_flag}"
        )

    def detailed_status(self) -> str:
        lines = [self.status_line(), "  Per-flag:"]
        for flag in KEEP_FLAGS:
            count = self.counts[flag]
            marker = " (full)" if count >= self.target_per_flag else ""
            lines.append(f"    {flag}: {count}/{self.target_per_flag}{marker}")
        return "\n".join(lines)

    def load_existing(self, output_path: Path) -> set[str]:
        """Load counts from existing output file. Returns set of processed RFP IDs."""
        processed: set[str] = set()
        if not output_path.exists():
            return processed
        with output_path.open() as f:
            for line in f:
                record = json.loads(line)
                processed.add(record["rfp_id"])
                flags = record.get("flags", [])
                if not flags:
                    self.no_flag_count += 1
                else:
                    for flag in flags:
                        if flag in self.counts:
                            self.counts[flag] += 1
        return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Label real RFP data with Claude")
    parser.add_argument("--source-dir", required=True, help="Path to RFP downloads directory")
    parser.add_argument("--target-per-flag", type=int, default=150)
    parser.add_argument("--target-no-flag", type=int, default=150)
    parser.add_argument("--output", default=str(DATA_DIR / "labeled_real.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Extract text only, no API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--max-rfps", type=int, default=0, help="Stop after N RFPs (0=unlimited)")
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output)
    source_dir = Path(args.source_dir)

    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        sys.exit(1)

    # Discover RFP directories
    rfp_dirs = sorted(d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    random.shuffle(rfp_dirs)
    print(f"Found {len(rfp_dirs)} RFP directories")

    # Initialize quota tracker
    tracker = QuotaTracker(args.target_per_flag, args.target_no_flag)
    processed_ids: set[str] = set()

    if args.resume and output_path.exists():
        processed_ids = tracker.load_existing(output_path)
        print(f"Resuming: {len(processed_ids)} RFPs already processed")
        print(f"  {tracker.status_line()}")

    if not args.dry_run:
        client = anthropic.AnthropicBedrock()

    # Open output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    extraction_failures = 0
    api_calls = 0
    start_time = time.time()

    with output_path.open(mode) as out_f:
        for i, rfp_dir in enumerate(rfp_dirs):
            rfp_id = rfp_dir.name

            if rfp_id in processed_ids:
                continue

            if tracker.all_filled():
                print(f"\nAll quotas filled after {i} RFPs!")
                break

            if args.max_rfps and (i + 1 - len(processed_ids)) > args.max_rfps:
                print(f"\nReached --max-rfps {args.max_rfps}")
                break

            # Progress
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(
                f"\n[{i + 1}/{len(rfp_dirs)}] {rfp_id} "
                f"({rate:.1f} RFPs/min) | {tracker.status_line()}"
            )

            # Extract text
            text = extract_rfp_text(rfp_dir)
            if not text.strip():
                print("    [skip] No extractable text")
                extraction_failures += 1
                continue

            word_count = len(text.split())
            print(f"    Extracted {word_count:,} words")

            if args.dry_run:
                continue

            # Split into sections if needed
            sections = split_into_sections(text)
            if len(sections) > 1:
                print(f"    Split into {len(sections)} sections")

            # Process each section
            rfp_chunks: list[dict] = []
            for sec_idx, section in enumerate(sections):
                needed = tracker.needed_flags()
                if not needed and tracker.no_flag_count >= tracker.target_no_flag:
                    break

                prompt = build_extraction_prompt(section, needed, tracker.filled_flags())
                chunks = call_claude(client, prompt)
                api_calls += 1

                # Validate and filter
                valid_chunks = []
                for chunk in chunks:
                    if not validate_chunk(chunk):
                        print(f"    [warn] Invalid chunk skipped in section {sec_idx}")
                        continue
                    valid_chunks.append(chunk)

                rfp_chunks.extend(valid_chunks)

            # Write results
            for ci, chunk in enumerate(rfp_chunks):
                record = {
                    "rfp_id": rfp_id,
                    "chunk_index": ci,
                    "chunk_text": chunk["chunk_text"],
                    "flags": chunk["flags"],
                    "confidence": chunk.get("confidence", "medium"),
                    "adversarial": chunk.get("adversarial", False),
                }
                out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            tracker.update(rfp_chunks)
            print(f"    Extracted {len(rfp_chunks)} chunks")

            # Canary: every 10 RFPs processed
            rfps_processed = i + 1 - len(processed_ids)
            if rfps_processed > 0 and rfps_processed % 10 == 0:
                print(f"\n--- Quota status (after {rfps_processed} new RFPs) ---")
                print(tracker.detailed_status())
                print("---")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed / 60:.1f} minutes")
    print(f"API calls: {api_calls}")
    print(f"Extraction failures: {extraction_failures}")
    print(f"Output: {output_path}")
    print(tracker.detailed_status())


if __name__ == "__main__":
    main()
