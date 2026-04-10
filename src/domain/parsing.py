from __future__ import annotations

from src.domain.taxonomy import VALID_FLAGS


def parse_flags(raw_output: str) -> list[str]:
    """Parse model output into a list of valid flag names.

    Strips whitespace, splits by newline, filters to known flags.
    Returns empty list for "no_flag" output or unparseable input.
    """
    if not raw_output or not raw_output.strip():
        return []

    lines = raw_output.strip().splitlines()
    flags = []
    for line in lines:
        name = line.strip()
        if name == "no_flag":
            continue
        if name in VALID_FLAGS:
            flags.append(name)

    return flags
