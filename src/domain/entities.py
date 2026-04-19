from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilterResult:
    filter: bool
    flags: set[str] = field(default_factory=set)
    unparsed_chunks: int = 0
    total_chunks: int = 0
