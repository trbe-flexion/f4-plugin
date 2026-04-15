#!/usr/bin/env python3
"""Clean zero-recall flags from opus_validated_real.jsonl.

Applies four cleaning operations:
  1. Strip brownfield flag entirely (no chunk-level signal)
  2. Strip budget_too_low from synthetic records and real records without dollar amounts
  3. Strip large_team from records without explicit headcount >=10 or 10+ role enumerations
  4. Strip onsite_required from records without explicit onsite keywords

Records that lose all flags get an empty flags list (treated as no_flag by
build_training_set.py). No records are deleted.

Usage:
  PYTHONPATH=. uv run python scripts/clean_zero_recall_flags.py
  PYTHONPATH=. uv run python scripts/clean_zero_recall_flags.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_FILE = REPO_ROOT / "data" / "opus_validated_real.jsonl"

# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

DOLLAR_RE = re.compile(r"\$\s*[\d,]+(?:\.\d+)?")

ONSITE_TERMS = [
    "onsite",
    "on-site",
    "on site",
    "in-person",
    "in person",
    "physically present",
    "physical presence",
    "place of performance",
]

HEADCOUNT_RE = re.compile(
    r"(?:"
    r"(?:1[0-9]|[2-9][0-9]|[1-9]\d{2,})\s*"
    r"(?:personnel|staff|fte|ftes|contractor|contractors|position|positions|"
    r"role|roles|people|employee|employees|worker|workers|team\s*member|team\s*members)"
    r"|"
    r"(?:personnel|staff|fte|ftes|contractor|contractors|position|positions|"
    r"role|roles|people|employee|employees|worker|workers|team\s*member|team\s*members)"
    r"\D{0,30}"
    r"(?:1[0-9]|[2-9][0-9]|[1-9]\d{2,})"
    r")",
    re.IGNORECASE,
)

ROLE_KEYWORDS = [
    "manager",
    "engineer",
    "analyst",
    "specialist",
    "lead",
    "architect",
    "developer",
    "sme",
    "administrator",
    "director",
    "technician",
    "coordinator",
    "programmer",
    "scientist",
    "advisor",
]


def has_dollar_amount(text: str) -> bool:
    return bool(DOLLAR_RE.search(text))


def has_explicit_headcount(text: str) -> bool:
    return bool(HEADCOUNT_RE.search(text))


def has_enumerated_roles(text: str, threshold: int = 10) -> bool:
    lines = text.split("\n")
    count = sum(1 for line in lines if any(kw in line.lower() for kw in ROLE_KEYWORDS))
    return count >= threshold


def has_large_team_signal(text: str) -> bool:
    return has_explicit_headcount(text) or has_enumerated_roles(text)


def has_onsite_signal(text: str) -> bool:
    text_lower = text.lower()
    return any(term in text_lower for term in ONSITE_TERMS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean zero-recall flags")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    records: list[dict] = []
    with DATA_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    brownfield_stripped = 0
    budget_stripped = 0
    large_team_stripped = 0
    onsite_stripped = 0

    for rec in records:
        flags = rec.get("flags", [])

        # 1. Strip brownfield entirely
        if "brownfield" in flags:
            flags.remove("brownfield")
            brownfield_stripped += 1

        # 2. Strip budget_too_low: synthetic or missing dollar amount
        if "budget_too_low" in flags:
            is_synthetic = rec.get("confidence") == "synthetic"
            if is_synthetic or not has_dollar_amount(rec["chunk_text"]):
                flags.remove("budget_too_low")
                budget_stripped += 1

        # 3. Strip large_team: no explicit headcount or role enumeration
        if "large_team" in flags and not has_large_team_signal(rec["chunk_text"]):
            flags.remove("large_team")
            large_team_stripped += 1

        # 4. Strip onsite_required: no explicit onsite keywords
        if "onsite_required" in flags and not has_onsite_signal(rec["chunk_text"]):
            flags.remove("onsite_required")
            onsite_stripped += 1

        rec["flags"] = flags

    print(f"brownfield stripped: {brownfield_stripped}")
    print(f"budget_too_low stripped: {budget_stripped}")
    print(f"large_team stripped: {large_team_stripped}")
    print(f"onsite_required stripped: {onsite_stripped}")

    # Count remaining
    from collections import Counter

    remaining: Counter = Counter()
    for rec in records:
        if rec.get("validation") == "dropped":
            continue
        for flag in rec.get("flags", []):
            remaining[flag] += 1

    print("\nRemaining flag counts (non-dropped):")
    for flag, count in remaining.most_common():
        print(f"  {flag}: {count}")

    if args.dry_run:
        print("\n(dry run — no files written)")
        return

    with DATA_FILE.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(records)} records to {DATA_FILE.name}")


if __name__ == "__main__":
    main()
