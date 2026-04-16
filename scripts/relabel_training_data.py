#!/usr/bin/env python3
"""Re-label training data from source RFPs using Opus.

Two-phase pipeline:
  Phase 1: Chunk each RFP (~400 words, 50 word overlap), label each chunk
           with Opus using the strict labeling prompt. Writes intermediary
           data/relabeled_pass1.jsonl.
  Phase 2: Validate flagged chunks — Opus reviews its own labels in a
           separate cognitive pass. Writes final data/train_relabel_2.jsonl.

Usage:
  PYTHONPATH=. uv run python scripts/relabel_training_data.py --phase 1 --dry-run
  PYTHONPATH=. uv run python scripts/relabel_training_data.py --phase 1 --concurrency 10
  PYTHONPATH=. uv run python scripts/relabel_training_data.py --phase 2 --concurrency 10
  PYTHONPATH=. uv run python scripts/relabel_training_data.py --phase both --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import anthropic

from src.frontend.extraction import extract_text

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
PASS1_FILE = DATA_DIR / "relabeled_pass1.jsonl"
FINAL_FILE = DATA_DIR / "train_relabel_2.jsonl"

DEFAULT_SOURCE = Path.home() / "library 2" / "downloads"

MODEL = "us.anthropic.claude-opus-4-6-v1"
CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 50

KEEP_FLAGS: list[str] = [
    "scope_misalignment",
    "waterfall_methodology",
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
    "onsite_madison",
    "brownfield",
    "large_team",
    "marginal_short_duration",
]

FLAG_DESCRIPTIONS: dict[str, str] = {
    "scope_misalignment": (
        "This chunk is a top-level scope statement — an introduction, overview, "
        "or purpose section that describes what the entire contract is for. The "
        "work described is clearly not custom software development. Do not flag "
        "task-level details, work breakdown items, technical requirements, or "
        "operational procedures, even if they describe non-software work. Only "
        "the chunk that answers 'what is this contract for?' should be flagged. "
        "Administrative sections, evaluation criteria, CLINs, and other "
        "non-scope content must never trigger this flag. A false positive "
        "permanently discards a revenue opportunity — only flag when confidence "
        "is very high."
    ),
    "waterfall_methodology": (
        "The chunk explicitly requires sequential/waterfall development as the "
        "methodology. Look for: 'waterfall,' 'traditional SDLC,' 'sequential "
        "phases,' 'fixed requirements baseline,' 'phase gate,' 'V-model.' "
        "Project phases, contract phases, and hardware installation sequences "
        "do NOT count — only actual waterfall SDLC mandates."
    ),
    "off_the_shelf_software": (
        "This chunk explicitly states what the procurement is for, and the "
        "primary deliverable is acquiring, licensing, or standing up a commercial "
        "off-the-shelf product — not building custom software. Ongoing support, "
        "maintenance, administration, or operations of commercial products "
        "already in use does not qualify. Only flag chunks that directly describe "
        "the overall purpose or scope of the contract. Administrative sections, "
        "task details, evaluation criteria, and other non-summary content must "
        "never trigger this flag, regardless of what they imply about the RFP. "
        "A false positive permanently discards a revenue opportunity — only flag "
        "when confidence is very high."
    ),
    "lpta_source_selection": (
        "Source selection for THIS solicitation is Lowest Price Technically "
        "Acceptable. Look for: 'LPTA,' 'lowest price technically acceptable.' "
        "The LPTA must apply to the evaluation of this RFP — do NOT flag "
        "references to LPTA for future task orders, add-ons, options, or other "
        "solicitations."
    ),
    "small_business_set_aside": (
        "This solicitation is set aside exclusively for small businesses. Look "
        "for: 'small business set-aside,' 'total small business set-aside,' "
        "FAR 52.219-6. The set-aside must be stated as applying to THIS "
        "solicitation — a FAR clause listed in an incorporated-by-reference "
        "table is NOT sufficient."
    ),
    "8a_set_aside": (
        "This solicitation is set aside under the 8(a) Business Development "
        "Program. Look for: '8(a) set-aside,' '8(a) sole source,' '8(a) "
        "competitive,' FAR 52.219-11/14. Must apply to THIS solicitation, "
        "not just listed in boilerplate."
    ),
    "wosb_set_aside": (
        "This solicitation is set aside for Women-Owned Small Businesses (WOSB) "
        "or Economically Disadvantaged WOSB (EDWOSB). Look for: 'WOSB set-aside,' "
        "'EDWOSB,' FAR 52.219-29/30. Must apply to THIS solicitation, not just "
        "listed in boilerplate."
    ),
    "sdvosb_set_aside": (
        "This solicitation is set aside for Service-Disabled Veteran-Owned Small "
        "Businesses. Look for: 'SDVOSB set-aside,' 'service-disabled veteran,' "
        "FAR 52.219-27. Must apply to THIS solicitation, not just listed in "
        "boilerplate."
    ),
    "hubzone_set_aside": (
        "This solicitation is set aside for HUBZone businesses. Look for: "
        "'HUBZone set-aside,' 'HUBZone price evaluation preference,' "
        "FAR 52.219-13. Must apply to THIS solicitation, not just listed "
        "in boilerplate."
    ),
    "agile_methodology": (
        "The chunk explicitly requires or describes Agile/Scrum methodology "
        "for the work. Look for: 'Agile,' 'Scrum,' 'sprint,' 'user stories,' "
        "'iterative development.' General mentions of modern practices or CI/CD "
        "alone are NOT sufficient — the chunk must specifically reference Agile "
        "as a methodology."
    ),
    "oral_presentation": (
        "The chunk describes a structured oral presentation or oral proposal as "
        "a distinct step in the evaluation/award process. Look for: 'oral "
        "presentation,' 'oral proposal,' 'offerors shall present.' The chunk "
        "must describe the oral presentation as an evaluation event — passing "
        "mentions of presentations in general context, interview logistics, or "
        "past performance narratives do NOT qualify. Flag even if the oral "
        "presentation is indicated to be optional."
    ),
    "design_exercise": (
        "The chunk describes a design challenge, prototype, proof of concept, or "
        "live demonstration as a distinct step in the evaluation/award process. "
        "Look for: 'design challenge,' 'challenge scenario,' 'proof of concept,' "
        "'POC,' 'flyoff,' 'pilot,' 'demonstration scenario.' The chunk must "
        "describe the exercise as an evaluation event — general mentions of "
        "prototyping or demos in the context of delivery methodology do NOT "
        "qualify. Flag even if the design exercise is indicated to be optional."
    ),
    "budget_too_low": (
        "The chunk states a total contract value, ceiling, or NTE under $100K "
        "as an explicit dollar figure. The dollar amount must clearly represent "
        "the overall monetary allocation for the entire contract (e.g., 'total "
        "contract value,' 'ceiling price,' 'not to exceed,' 'total estimated "
        "cost'). Do NOT flag: CLIN line items, individual option year values, "
        "travel budgets, subcontract amounts, funding increments, per-unit or "
        "hourly rates, dollar thresholds in policy/reporting rules, or other "
        "partial figures. When in doubt, do not flag. A false positive "
        "permanently discards a revenue opportunity — only flag when confidence "
        "is very high."
    ),
    "onsite_required": (
        "The chunk explicitly mandates that all or substantially all work must "
        "be performed at a specific physical location. Look for: 'on-site,' "
        "'onsite,' 'in-person,' 'physically present,' 'work shall be performed "
        "at.' A 'place of performance' section header or address alone is NOT "
        "sufficient — the chunk must state that onsite presence is required as "
        "a condition of the work. Do NOT flag if remote, hybrid, or telework "
        "options are mentioned. Do NOT flag if the location is Madison, WI "
        "(use onsite_madison instead). A false positive permanently discards "
        "a revenue opportunity — only flag when confidence is very high."
    ),
    "onsite_madison": (
        "The chunk explicitly mandates onsite work AND specifies Madison, WI "
        "(or Madison, Wisconsin) as the location. Both conditions must be "
        "present in the chunk: an explicit onsite mandate + Madison, WI. A "
        "'place of performance' header or address alone is NOT sufficient — "
        "the chunk must state that onsite presence is required. If onsite is "
        "required but the location is not Madison, use onsite_required instead."
    ),
    "brownfield": (
        "The contractor will inherit and work within an existing codebase "
        "developed by a prior team. The chunk must make clear that the "
        "contractor is expected to take over, maintain, or extend running "
        "software — not build a replacement. Generic terms like 'modernize,' "
        "'transition,' or 'legacy' are NOT sufficient on their own, as they "
        "appear frequently in non-brownfield contexts. The chunk must contain "
        "strong, specific evidence that existing code is being handed to the "
        "contractor. When in doubt, do not flag."
    ),
    "large_team": (
        "The chunk indicates the contractor's team must include 10 or more "
        "personnel or FTEs. Must be an explicit count of the contractor's own "
        "team — not user counts, end-user populations, vendor counts, government "
        "staff, or site/location counts. The number must clearly refer to the "
        "contractor's staffing requirement. When in doubt, do not flag."
    ),
    "marginal_short_duration": (
        "The total period of performance for the contract is under 12 months. "
        "Must be a stated total duration, not an individual option period, task "
        "order duration, phase duration, or transition period within a longer "
        "contract. 12 months exactly does NOT qualify. When in doubt, do not flag."
    ),
}

KEEP_FLAG_SET = set(KEEP_FLAGS)


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
    """Extract text from all PDF/DOCX files in a directory."""
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
# Pass 1 prompt (strict chunk-level labeling)
# ---------------------------------------------------------------------------


def build_labeling_prompt(chunk: str) -> str:
    flag_block = "\n".join(f"  - {flag}: {FLAG_DESCRIPTIONS[flag]}" for flag in KEEP_FLAGS)
    return f"""Your output will be parsed directly by a script. You MUST return ONLY
a JSON object in this exact format, with no other text:
{{"flags": ["flag_name_1", "flag_name_2"]}}
or if no flags: {{"flags": []}}

## Context

You are labeling government RFP text chunks for Flexion, a software consultancy
that screens RFPs to decide whether to bid. Each flag represents a specific
business signal that affects the bid/no-bid decision. These labels are ground
truth for a fine-tuned detection model that replaces simple string matching.
The model must be smarter than keyword search in both directions: catching real
signals that string matching would miss, and ignoring boilerplate that string
matching would falsely trigger on. Flag with confidence when the evidence is
clear, but do not flag on weak or ambiguous evidence.

You are seeing one ~400-word chunk at a time, not the full document. You must
decide based on what is present in this chunk, not inferred from context clues,
document titles, boilerplate, or indirect language.

## Flags to detect:
{flag_block}

## Rules:
- Only flag what is present in the chunk text, not inferred from indirect clues or boilerplate
- Consider each flag independently. For each flag, ask: does this chunk contain \
direct evidence for this specific flag? Do not let the presence of one flag \
influence your judgment of others.
- When in doubt about a specific flag, do not apply it
- Boilerplate, contract clauses, and legal terms are almost never flaggable
- It is perfectly acceptable for a chunk to have no flags
- Your output MUST return ONLY a JSON object in this exact format, with no other text: \
{{"flags": ["flag_name_1", "flag_name_2"]}} or if no flags: {{"flags": []}}

## Chunk:
{chunk}

Return ONLY: {{"flags": ["flag_name", ...]}} or {{"flags": []}}"""


# ---------------------------------------------------------------------------
# Pass 2 prompt (validation of flagged chunks)
# ---------------------------------------------------------------------------


def build_validation_prompt(chunk: str, flags: list[str]) -> str:
    BLACK_FLAGS = {
        "scope_misalignment",
        "onsite_required",
        "off_the_shelf_software",
        "budget_too_low",
    }

    flag_lines = []
    for flag in flags:
        if flag not in FLAG_DESCRIPTIONS:
            continue
        prefix = "[BLACK — fast-fail] " if flag in BLACK_FLAGS else ""
        flag_lines.append(f"  - {prefix}{flag}: {FLAG_DESCRIPTIONS[flag]}")
    flag_block = "\n".join(flag_lines)

    flags_str = ", ".join(flags)
    return f"""CRITICAL: Return ONLY a JSON object. No reasoning, no explanation, no
preamble. Your entire response must be parseable as JSON.

Format: {{"flags": ["flag_name_1"]}} or {{"flags": []}}

---

Validate these flags against the chunk below. Remove any flag that lacks
clear, direct textual evidence per its definition. Keep only flags with
explicit support in the chunk text.

Flags to validate: {flags_str}

Definitions:
{flag_block}

Removal criteria (apply strictly):
- No specific text in the chunk satisfies the definition → remove
- BLACK flags are fast-fail (auto-disqualify the RFP). False positives
  permanently discard revenue opportunities. Remove unless unambiguous.
- Evidence is in an administrative section (eval criteria, CLINs, contract
  clauses, SF 1449 forms, FAR provisions, continuation sheets) → remove
- Evidence is indirect, boilerplate, headers, or context clues → remove
- You may ONLY keep flags from the original list — do not add new flags

Chunk:
{chunk}

Respond with ONLY: {{"flags": [...]}}"""


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------


async def call_opus(
    client: anthropic.AsyncAnthropicBedrock,
    prompt: str,
    semaphore: asyncio.Semaphore,
    label: str,
) -> list[str]:
    """Call Opus and parse JSON response. Returns list of flag names."""
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

                # Extract JSON object from anywhere in the response
                brace_start = content.find("{")
                brace_end = content.rfind("}")
                if brace_start != -1 and brace_end > brace_start:
                    content = content[brace_start : brace_end + 1]

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
                print(f"         raw: {content[:200]}")
                return []
            except Exception as e:
                print(f"  [error] {label}: {e}")
                return []
    print(f"  [fail] {label}: exhausted retries")
    return []


# ---------------------------------------------------------------------------
# Phase 1: Chunk + Label
# ---------------------------------------------------------------------------


async def run_phase1(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        return

    # Discover RFP directories
    skip_names = {".DS_Store", "desktop.ini"}
    rfp_dirs = sorted(
        d
        for d in source_dir.iterdir()
        if d.is_dir() and d.name not in skip_names and not d.name.startswith(".")
    )
    print(f"Found {len(rfp_dirs)} RFP directories")

    # Limit extraction to --max-rfps if set (avoids extracting all 958 RFPs
    # when only a quick spot-check is needed)
    if args.max_rfps:
        rfp_dirs = rfp_dirs[: args.max_rfps]
        print(f"  Limited to first {args.max_rfps} RFPs for extraction")

    # Extract and chunk RFPs
    all_chunks: list[tuple[str, str, int]] = []  # (rfp_id, chunk_text, chunk_idx)
    for rfp_dir in rfp_dirs:
        rfp_id = rfp_dir.name
        text = extract_rfp_text(rfp_dir)
        if not text:
            print(f"  [skip] {rfp_id}: no text extracted")
            continue
        words = len(text.split())
        chunks = chunk_text(text)
        print(f"  {rfp_id}: {words} words, {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append((rfp_id, chunk, i))

    print(f"\nTotal chunks to label: {len(all_chunks)}")

    if args.dry_run:
        print("(dry run — no API calls)")
        return

    # Resume support: load already-labeled chunks
    PASS1_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_keys: set[tuple[str, int]] = set()
    if PASS1_FILE.exists():
        with PASS1_FILE.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_keys.add((rec["rfp_id"], rec["chunk_index"]))
        if done_keys:
            print(f"  Resuming: {len(done_keys)} chunks already labeled")

    remaining = [(rid, t, i) for rid, t, i in all_chunks if (rid, i) not in done_keys]

    print(f"  Chunks to label: {len(remaining)}")

    if not remaining:
        print("  All chunks already labeled.")
        _print_pass1_summary()
        return

    # Label concurrently
    client = anthropic.AsyncAnthropicBedrock()
    semaphore = asyncio.Semaphore(args.concurrency)

    flag_counts: Counter = Counter()
    no_flag_count = 0
    flagged_count = 0
    labeled_count = len(done_keys)
    rfps_seen: set[str] = set()
    total_rfps = len({rid for rid, _, _ in remaining})

    # Count existing flagged chunks for --max-flagged tracking
    if PASS1_FILE.exists() and done_keys:
        with PASS1_FILE.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec["flags"]:
                        flagged_count += 1

    outfile = PASS1_FILE.open("a")
    stop_event = asyncio.Event()

    async def label_chunk(rfp_id: str, text: str, idx: int) -> None:
        nonlocal labeled_count, no_flag_count, flagged_count
        if stop_event.is_set():
            return
        call_label = f"{rfp_id}[{idx}]"
        prompt = build_labeling_prompt(text)
        flags = await call_opus(client, prompt, semaphore, call_label)
        if stop_event.is_set():
            return
        flags = [f for f in flags if f in KEEP_FLAG_SET]

        rec = {
            "rfp_id": rfp_id,
            "chunk_index": idx,
            "chunk_text": text,
            "flags": flags,
        }
        outfile.write(json.dumps(rec) + "\n")
        outfile.flush()
        labeled_count += 1
        rfps_seen.add(rfp_id)

        if flags:
            for flag in flags:
                flag_counts[flag] += 1
            flagged_count += 1
            if args.max_flagged and flagged_count >= args.max_flagged:
                print(f"\n  Reached --max-flagged {args.max_flagged}, stopping.")
                stop_event.set()
        else:
            no_flag_count += 1

        if labeled_count % 100 == 0:
            print(
                f"  Progress: {labeled_count} chunks labeled "
                f"({flagged_count} flagged) | "
                f"{len(rfps_seen)}/{total_rfps} RFPs"
            )

    # Process in batches to allow early stop
    batch_size = args.concurrency * 2
    for batch_start in range(0, len(remaining), batch_size):
        if stop_event.is_set():
            break
        batch = remaining[batch_start : batch_start + batch_size]
        await asyncio.gather(*[label_chunk(rid, text, idx) for rid, text, idx in batch])
    outfile.close()

    _print_pass1_summary()


def _print_pass1_summary() -> None:
    """Print summary stats from the Pass 1 file."""
    flag_counts: Counter = Counter()
    no_flag = 0
    total = 0
    with PASS1_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if rec["flags"]:
                for flag in rec["flags"]:
                    flag_counts[flag] += 1
            else:
                no_flag += 1

    flagged = total - no_flag
    print(f"\nPass 1 summary ({PASS1_FILE.name}):")
    print(f"  Total: {total} chunks | Flagged: {flagged} | No-flag: {no_flag}")
    print("  Flag distribution:")
    for flag, count in flag_counts.most_common():
        print(f"    {flag}: {count}")


# ---------------------------------------------------------------------------
# Phase 2: Validate flagged chunks
# ---------------------------------------------------------------------------


async def run_phase2(args: argparse.Namespace) -> None:
    if not PASS1_FILE.exists():
        print(f"Error: Pass 1 file not found: {PASS1_FILE}")
        print("  Run --phase 1 first.")
        return

    # Load all Pass 1 records
    records: list[dict] = []
    with PASS1_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {PASS1_FILE.name}")

    flagged = [r for r in records if r["flags"]]
    no_flag = [r for r in records if not r["flags"]]
    print(f"  Flagged: {len(flagged)} | No-flag: {len(no_flag)}")

    if args.dry_run:
        print("(dry run — no API calls)")
        return

    # Resume support: count already-validated records
    FINAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    already_done = 0
    if FINAL_FILE.exists():
        with FINAL_FILE.open() as f:
            for line in f:
                if line.strip():
                    already_done += 1
        if already_done:
            print(f"  Resuming: {already_done} records already validated")

    # Process in order: no-flag chunks pass through, flagged get validated
    if already_done >= len(records):
        print("  All records already validated.")
        _print_phase2_summary()
        return

    remaining = records[already_done:]
    print(f"  Records to process: {len(remaining)}")

    client = anthropic.AsyncAnthropicBedrock()
    semaphore = asyncio.Semaphore(args.concurrency)

    kept_count = 0
    fixed_count = 0
    skip_count = 0
    processed = already_done

    outfile = FINAL_FILE.open("a")

    async def validate_record(rec: dict) -> dict:
        """Validate a single record. No-flag records pass through."""
        if not rec["flags"]:
            return {**rec, "validation": "skip"}

        call_label = f"{rec['rfp_id']}[{rec['chunk_index']}]"
        prompt = build_validation_prompt(rec["chunk_text"], rec["flags"])
        validated_flags = await call_opus(client, prompt, semaphore, f"v:{call_label}")
        validated_flags = [f for f in validated_flags if f in KEEP_FLAG_SET]

        # Only keep flags that were in the original set (no new additions)
        original_set = set(rec["flags"])
        validated_flags = [f for f in validated_flags if f in original_set]

        if set(validated_flags) == original_set:
            return {**rec, "validation": "keep"}
        else:
            return {
                **rec,
                "original_flags": rec["flags"],
                "flags": validated_flags,
                "validation": "fixed",
            }

    # Process in batches to maintain order for resume support
    batch_size = args.concurrency * 2
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]
        results = await asyncio.gather(*[validate_record(r) for r in batch])

        for result in results:
            outfile.write(json.dumps(result) + "\n")
            processed += 1

            if result["validation"] == "skip":
                skip_count += 1
            elif result["validation"] == "keep":
                kept_count += 1
            elif result["validation"] == "fixed":
                fixed_count += 1

        outfile.flush()

        if processed % 500 == 0 or batch_start + batch_size >= len(remaining):
            print(
                f"  Progress: {processed}/{len(records)} | "
                f"keep={kept_count} fix={fixed_count} skip={skip_count}"
            )

    outfile.close()

    _print_phase2_summary()


def _print_phase2_summary() -> None:
    """Print summary stats from the final validated file."""
    flag_counts: Counter = Counter()
    no_flag = 0
    total = 0
    kept = 0
    fixed = 0
    skipped = 0
    flags_removed: Counter = Counter()
    with FINAL_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            v = rec.get("validation", "")
            if v == "keep":
                kept += 1
            elif v == "fixed":
                fixed += 1
                orig = set(rec.get("original_flags", []))
                curr = set(rec["flags"])
                for f_name in orig - curr:
                    flags_removed[f_name] += 1
            elif v == "skip":
                skipped += 1

            if rec["flags"]:
                for flag in rec["flags"]:
                    flag_counts[flag] += 1
            else:
                no_flag += 1

    flagged = total - no_flag
    print(f"\nPhase 2 summary ({FINAL_FILE.name}):")
    print(f"  Total: {total} | Flagged: {flagged} | No-flag: {no_flag}")
    print(f"  Verdicts: keep={kept} fixed={fixed} skip(no-flag)={skipped}")
    if flags_removed:
        print("  Flags removed by validation:")
        for f_name, count in flags_removed.most_common():
            print(f"    {f_name}: -{count}")
    print("  Final flag distribution:")
    for flag, count in flag_counts.most_common():
        print(f"    {flag}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-label training data from source RFPs using Opus"
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "both"],
        required=True,
        help="Which phase to run: 1 (label), 2 (validate), both",
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE),
        help=f"Path to RFP downloads directory (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent Opus calls (default: 10)",
    )
    parser.add_argument(
        "--max-rfps",
        type=int,
        default=0,
        help="Phase 1 only: stop after N RFPs (0=unlimited)",
    )
    parser.add_argument(
        "--max-flagged",
        type=int,
        default=0,
        help="Phase 1 only: stop after N flagged (non-no_flag) chunks (0=unlimited)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without calling API",
    )
    args = parser.parse_args()

    if args.phase in ("1", "both"):
        print("=" * 60)
        print("PHASE 1: Chunk + Label")
        print("=" * 60)
        asyncio.run(run_phase1(args))

    if args.phase in ("2", "both"):
        print("\n" + "=" * 60)
        print("PHASE 2: Validate flagged chunks")
        print("=" * 60)
        asyncio.run(run_phase2(args))


if __name__ == "__main__":
    main()
