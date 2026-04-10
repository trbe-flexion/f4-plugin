from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FlagDetector(Protocol):
    """Port for model inference. Returns raw model output string."""

    def detect_flags(self, chunk: str) -> str: ...
