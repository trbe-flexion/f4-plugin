#!/usr/bin/env python3
"""Generate synthetic training data for F4 flag detection fine-tuning.

Produces:
  data/rag_seeds.jsonl  — synthetic RAG seed passages per flag
  data/train.jsonl      — 80% training split (chat messages format)
  data/eval.jsonl       — 20% eval split (chat messages format)

Usage:
  uv run scripts/generate_data.py

Requires AWS credentials in environment with bedrock:InvokeModel permission.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
SYSTEM_PROMPT_PATH = REPO_ROOT / ".development-notes" / "system-prompt.md"

MODEL = "us.anthropic.claude-sonnet-4-6"
EXAMPLES_PER_FLAG = 100  # positive + adversarial per flag
NO_FLAG_EXAMPLES = 400
RAG_SEEDS_PER_FLAG = 10
TRAIN_SPLIT = 0.8
BATCH_SIZE = 5

# Reproducibility
random.seed(42)

# ---------------------------------------------------------------------------
# Flag definitions
# ---------------------------------------------------------------------------

KEEP_FLAGS: dict[str, str] = {
    "waterfall_methodology": (
        'Explicitly requires sequential/waterfall development. Look for: "waterfall," '
        '"traditional SDLC," "sequential phases," "standard software lifecycle development," '
        '"fixed requirements baseline," "phase gate," "V-model."'
    ),
    "off_the_shelf_software": (
        "Primary work is configuring or deploying commercial off-the-shelf platforms. "
        "Look for product names (Salesforce, SharePoint, WordPress, Wix, Squarespace) or COTS "
        'language ("off-the-shelf," "commercial solution," "configure and deploy," '
        '"pre-built platform").'
    ),
    "no_custom_development": (
        "Explicitly excludes custom software development. Look for language in scope or "
        'exclusions: "no custom development," "no software development," '
        '"configuration only," "COTS solution required."'
    ),
    "lpta_source_selection": (
        'Source selection is Lowest Price Technically Acceptable. Look for: "LPTA," '
        '"lowest price technically acceptable."'
    ),
    "small_business_set_aside": ("RFP is set aside exclusively for small businesses."),
    "8a_set_aside": ("8(a) Business Development Program set-aside."),
    "wosb_set_aside": "Women-Owned Small Business (WOSB) set-aside.",
    "edwosb_set_aside": "Economically Disadvantaged Women-Owned Small Business (EDWOSB) set-aside.",
    "sdvosb_set_aside": "Service-Disabled Veteran-Owned Small Business (SDVOSB) set-aside.",
    "hubzone_set_aside": "Historically Underutilized Business Zone (HUBZone) set-aside.",
    "agile_methodology": ("Explicitly requires or expects Agile/Scrum methodology."),
    "oral_presentation": ("Includes an oral presentation component as part of evaluation."),
    "design_exercise": ("Includes a design challenge, prototype submission, or hands-on exercise."),
    "budget_too_low": "Total contract budget is below $100K.",
    "brownfield": (
        "Contractor is taking over an existing codebase or continuing work from a prior team. "
        'Look for: "existing system," "transition from incumbent," '
        '"continuing work," "legacy codebase."'
    ),
    "onsite_required": (
        "All work must be performed at a specific location (not Madison, WI). "
        "Do not detect if hybrid, remote, or flexible options are offered."
    ),
    "onsite_madison": ("Onsite work is required and the location is Madison, WI specifically."),
    "large_team": "Scope requires 10 or more personnel.",
    "marginal_short_duration": "Period of performance is less than 12 months.",
}

# Multi-flag combinations to generate examples for
MULTI_FLAG_PAIRS: list[list[str]] = [
    ["waterfall_methodology", "onsite_required"],
    ["lpta_source_selection", "small_business_set_aside"],
    ["off_the_shelf_software", "no_custom_development"],
    ["agile_methodology", "oral_presentation"],
    ["brownfield", "marginal_short_duration"],
    ["8a_set_aside", "oral_presentation"],
    ["onsite_madison", "large_team"],
    ["agile_methodology", "design_exercise"],
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def rag_seed_prompt(flag: str, definition: str, n: int) -> str:
    return f"""Generate {n} realistic government RFP text passages (150-300 words each)
that clearly and explicitly exhibit the following flag.

Flag: {flag}
Definition: {definition}

Requirements:
- Write in authentic federal government solicitation language
- The flag must be clearly and explicitly present — quote exact trigger phrases where relevant
- Each passage should be distinct in agency, program type, and phrasing
- Do not include any flag labels or annotations in the passages

Return a JSON array of {n} strings. No other text."""


def positive_examples_prompt(flag: str, definition: str, seeds: list[str], n: int) -> str:
    rag_block = "\n\n".join(f"Example:\n{ex}" for ex in seeds[:3])
    return f"""Generate {n} realistic government RFP text chunks (200-400 words each)
for flag detection training.

Target flag: {flag}
Definition: {definition}

Reference examples of authentic RFP language for this flag:
{rag_block}

Requirements:
- Write in authentic federal government solicitation language
- The flag must be clearly and explicitly present in each chunk
- Vary agency type, program context, and phrasing across chunks
- Each chunk should read as a natural excerpt from a larger RFP
- Do not include flag labels or annotations

Return a JSON array of {n} strings. No other text."""


def adversarial_examples_prompt(flag: str, definition: str, n: int) -> str:
    return f"""Generate {n} realistic government RFP text chunks (200-400 words each)
where the flag "{flag}" should NOT be detected, despite containing related language.

Flag definition: {definition}

Adversarial patterns to vary across examples:
- Mention the concept but explicitly negate it (e.g., "not required to work on-site")
- Use the terminology in a context where it does not apply
- Describe a related but distinct concept
- Include qualifying language that changes the meaning

Requirements:
- Write in authentic federal government solicitation language
- The flag must clearly NOT apply on close reading
- Do not include flag labels or annotations

Return a JSON array of {n} strings. No other text."""


def no_flag_prompt(n: int) -> str:
    return f"""Generate {n} realistic government RFP text chunks (200-400 words each)
that contain none of the following flags:

waterfall_methodology, off_the_shelf_software, no_custom_development, lpta_source_selection,
small_business_set_aside, 8a_set_aside, wosb_set_aside, edwosb_set_aside, sdvosb_set_aside,
hubzone_set_aside, agile_methodology, oral_presentation, design_exercise, budget_too_low,
brownfield, onsite_required, onsite_madison, large_team, marginal_short_duration

Requirements:
- Write in authentic federal government solicitation language
- Describe custom software development opportunities with no disqualifying or notable signals
- Vary agencies, program types, technology areas, and contract structures
- Do not include flag labels or annotations

Return a JSON array of {n} strings. No other text."""


def multi_flag_prompt(flags: list[str], n: int) -> str:
    flag_block = "\n".join(f"- {f}: {KEEP_FLAGS[f]}" for f in flags)
    return f"""Generate {n} realistic government RFP text chunks (200-400 words each)
that exhibit ALL of the following flags simultaneously.

Flags:
{flag_block}

Requirements:
- Write in authentic federal government solicitation language
- All listed flags must be clearly and explicitly present in each chunk
- Each chunk should read as a natural excerpt from a larger RFP
- Do not include flag labels or annotations

Return a JSON array of {n} strings. No other text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text().strip()


def simulate_rag_context(flag: str, seeds: list[str], top_k: int = 3) -> str:
    selected = random.sample(seeds, min(top_k, len(seeds)))
    parts = ["[Retrieved context]\n"]
    for i, passage in enumerate(selected, 1):
        parts.append(f"Example {i}:\n{passage}\n")
    return "\n".join(parts)


def build_user_message(chunk: str, rag_context: str) -> str:
    return f"{rag_context}\n---\n\nRFP chunk to analyze:\n{chunk}"


def call_claude(client: anthropic.AnthropicBedrock, prompt: str, batch_size: int = 5) -> list[str]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.content[0].text.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[1:end])
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"    [warn] JSON parse failed (batch={batch_size}), tail: {content[-200:]}")
        if batch_size > 1:
            # Response was truncated — split into two smaller calls
            half = batch_size // 2
            print(f"    [retry] splitting into batches of {half}")
            # Re-derive a smaller prompt by replacing the batch count in the original
            smaller_prompt = prompt.replace(f"Generate {batch_size} ", f"Generate {half} ", 1)
            a = call_claude(client, smaller_prompt, half)
            b = call_claude(client, smaller_prompt, half)
            return a + b
        raise


def generate_in_batches(
    client: anthropic.AnthropicBedrock,
    prompt_fn,
    total: int,
) -> list[str]:
    results: list[str] = []
    while len(results) < total:
        batch = min(BATCH_SIZE, total - len(results))
        results.extend(call_claude(client, prompt_fn(batch)))
    return results[:total]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(records)} records → {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Generation phases
# ---------------------------------------------------------------------------


def generate_rag_seeds(client: anthropic.AnthropicBedrock) -> dict[str, list[str]]:
    print("\n=== Phase 1: RAG seeds ===")
    seeds: dict[str, list[str]] = {}
    for flag, definition in KEEP_FLAGS.items():
        print(f"  {flag}")
        seeds[flag] = call_claude(client, rag_seed_prompt(flag, definition, RAG_SEEDS_PER_FLAG))
    return seeds


PARTIAL_FILE = DATA_DIR / "examples_partial.jsonl"


def load_completed_sections() -> tuple[set[str], list[dict]]:
    """Return set of completed section keys and all examples written so far."""
    completed: set[str] = set()
    examples: list[dict] = []
    if PARTIAL_FILE.exists():
        with PARTIAL_FILE.open() as f:
            for line in f:
                record = json.loads(line)
                completed.add(record["_section"])
                examples.append(record["example"])
    return completed, examples


def append_section(section: str, new_examples: list[dict]) -> None:
    PARTIAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PARTIAL_FILE.open("a") as f:
        for ex in new_examples:
            f.write(json.dumps({"_section": section, "example": ex}) + "\n")


def generate_examples(
    client: anthropic.AnthropicBedrock,
    seeds: dict[str, list[str]],
    system_prompt: str,
) -> list[dict]:
    print("\n=== Phase 2: Training examples ===")
    completed, all_examples = load_completed_sections()
    if completed:
        print(f"  Resuming — {len(completed)} sections done, {len(all_examples)} examples loaded")
    flag_list = list(KEEP_FLAGS.keys())

    def make_example(chunk: str, rag_ctx: str, output: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(chunk, rag_ctx)},
                {"role": "assistant", "content": output},
            ]
        }

    # Per-flag: positive + adversarial
    for flag, definition in KEEP_FLAGS.items():
        if flag in completed:
            print(f"  {flag} (skipped)")
            continue
        print(f"  {flag}")
        flag_seeds = seeds[flag]
        flag_examples: list[dict] = []

        n_positive = int(EXAMPLES_PER_FLAG * 0.85)
        chunks = generate_in_batches(
            client,
            lambda n, f=flag, d=definition, s=flag_seeds: positive_examples_prompt(f, d, s, n),
            n_positive,
        )
        for chunk in chunks:
            flag_examples.append(make_example(chunk, simulate_rag_context(flag, flag_seeds), flag))

        n_adversarial = EXAMPLES_PER_FLAG - n_positive
        adv_chunks = generate_in_batches(
            client,
            lambda n, f=flag, d=definition: adversarial_examples_prompt(f, d, n),
            n_adversarial,
        )
        for chunk in adv_chunks:
            flag_examples.append(
                make_example(chunk, simulate_rag_context(flag, flag_seeds), "no_flag")
            )

        append_section(flag, flag_examples)
        all_examples.extend(flag_examples)

    # Multi-flag examples
    if "multi_flag" not in completed:
        print("  multi-flag combinations")
        multi_examples: list[dict] = []
        for pair in MULTI_FLAG_PAIRS:
            chunks = call_claude(client, multi_flag_prompt(pair, BATCH_SIZE))
            for chunk in chunks:
                rag_ctx = simulate_rag_context(pair[0], seeds[pair[0]], top_k=2)
                multi_examples.append(make_example(chunk, rag_ctx, "\n".join(pair)))
        append_section("multi_flag", multi_examples)
        all_examples.extend(multi_examples)
    else:
        print("  multi-flag combinations (skipped)")

    # No-flag examples
    if "no_flag" not in completed:
        print("  no_flag")
        no_flag_examples: list[dict] = []
        no_flag_chunks = generate_in_batches(
            client,
            lambda n: no_flag_prompt(n),
            NO_FLAG_EXAMPLES,
        )
        for chunk in no_flag_chunks:
            random_flag = random.choice(flag_list)
            rag_ctx = simulate_rag_context(random_flag, seeds[random_flag])
            no_flag_examples.append(make_example(chunk, rag_ctx, "no_flag"))
        append_section("no_flag", no_flag_examples)
        all_examples.extend(no_flag_examples)
    else:
        print("  no_flag (skipped)")

    return all_examples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_seeds_from_file() -> dict[str, list[str]]:
    seeds: dict[str, list[str]] = {}
    with (DATA_DIR / "rag_seeds.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            seeds.setdefault(record["flag"], []).append(record["passage"])
    return seeds


def main() -> None:
    client = anthropic.AnthropicBedrock()
    system_prompt = load_system_prompt()

    seeds_path = DATA_DIR / "rag_seeds.jsonl"
    if seeds_path.exists():
        print("\n=== Phase 1: RAG seeds (loading from file) ===")
        seeds = load_seeds_from_file()
        print(f"  Loaded {sum(len(v) for v in seeds.values())} seeds for {len(seeds)} flags")
    else:
        seeds = generate_rag_seeds(client)
        write_jsonl(
            seeds_path,
            [{"flag": flag, "passage": p} for flag, passages in seeds.items() for p in passages],
        )

    examples = generate_examples(client, seeds, system_prompt)
    print(f"\nTotal examples: {len(examples)}")

    random.shuffle(examples)
    split = int(len(examples) * TRAIN_SPLIT)
    write_jsonl(DATA_DIR / "train.jsonl", examples[:split])
    write_jsonl(DATA_DIR / "eval.jsonl", examples[split:])

    if PARTIAL_FILE.exists():
        PARTIAL_FILE.unlink()
        print(f"  Deleted {PARTIAL_FILE.relative_to(REPO_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
