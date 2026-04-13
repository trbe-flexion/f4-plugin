from __future__ import annotations

import re

from src.domain.taxonomy import VALID_FLAGS


def parse_flags(raw_output: str) -> list[str]:
    """Parse model output into a list of valid flag names.

    Handles variations: newline-separated, comma-separated, mixed,
    and 'none'/'None' as aliases for no_flag. Extracts valid flag
    names even if surrounded by prose.
    """
    if not raw_output or not raw_output.strip():
        return []

    text = raw_output.strip()

    if text.lower() in ("no_flag", "none"):
        return []

    # Split on newlines and commas
    tokens = re.split(r"[\n,]+", text)

    flags = []
    for token in tokens:
        name = token.strip()
        if name in VALID_FLAGS:
            flags.append(name)

    return flags
