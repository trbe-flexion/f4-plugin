from __future__ import annotations

from src.domain.taxonomy import BLACK_FLAGS, RED_FLAGS


class FilterDecisionEngine:
    """Determines whether an RFP should be filtered based on detected flags."""

    def __init__(self, red_flag_threshold: int = 999):
        self.red_flag_threshold = red_flag_threshold

    def decide(self, flags: set[str]) -> bool:
        """Return True if the RFP should be filtered out.

        Filters if any black flag is present, or if the number of
        red flags meets or exceeds the threshold.
        """
        if flags & BLACK_FLAGS:
            return True

        red_count = len(flags & RED_FLAGS)
        return red_count >= self.red_flag_threshold
