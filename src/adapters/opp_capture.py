"""F4 adapter for opp-capture's Analyst pipeline.

This module is the integration seam between F4 and opp-capture. It
encapsulates all F4 internals so opp-capture only needs to import,
optionally configure, and call .filter().

Usage in opp-capture's Analyst.analyze_and_evaluate(), after text
extraction and BEFORE the summarization LLM call:

    from src.adapters.opp_capture import F4Adapter

    f4 = F4Adapter(model_id="arn:aws:bedrock:us-east-1:...")

    # In the per-opportunity loop, after combined_text is built:
    f4_result = f4.filter(combined_text)

    # Short-circuit on black flags (fast-fail before any LLM cost)
    if f4_result["black"]:
        decision = Decision(
            opportunity_id=opportunity_id,
            title=opportunity.title or "Unknown",
            decision=DecisionType.FAIL,
            reasoning=f"F4 fast-fail: {', '.join(f4_result['black'])}",
            fast_fail_criteria_triggered=f4_result["black"],
            red_flags=f4_result["red"],
            green_flags=f4_result["green"],
            blue_flags=f4_result["blue"],
            ...
        )
        decision_writer.write(decision)
        continue

    # Otherwise, run Sonnet summarization + evaluation as usual, then merge:
    decision = replace(
        decision,
        red_flags=list(set(decision.red_flags + f4_result["red"])),
        green_flags=list(set(decision.green_flags + f4_result["green"])),
        blue_flags=list(set(decision.blue_flags + f4_result["blue"])),
    )
"""

from __future__ import annotations

from src.domain.taxonomy import BLACK_FLAGS, BLUE_FLAGS, GREEN_FLAGS, RED_FLAGS
from src.inference.bedrock import BedrockFlagDetector
from src.pipeline.filter import F4Pipeline


class F4Adapter:
    """Public API for opp-capture integration.

    Import, optionally configure, call .filter(). All F4 internals
    (chunking, RAG, inference, parsing, decision logic) are encapsulated.
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        max_workers: int = 4,
        temperature: float = 0.1,
        red_flag_threshold: int = 999,
    ) -> None:
        detector = BedrockFlagDetector(
            model_id=model_id,
            region=region,
            temperature=temperature,
        )
        self._pipeline = F4Pipeline(
            flag_detector=detector,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            max_workers=max_workers,
            red_flag_threshold=red_flag_threshold,
        )

    def filter(self, text: str) -> dict[str, list[str] | int]:
        """Run F4 on raw document text and return flags grouped by color.

        Returns:
            {
                "black": [...],
                "red": [...],
                "green": [...],
                "blue": [...],
                "unparsed_chunks": int,
                "total_chunks": int,
            }
        """
        result = self._pipeline.filter(text)

        return {
            "black": sorted(result.flags & BLACK_FLAGS),
            "red": sorted(result.flags & RED_FLAGS),
            "green": sorted(result.flags & GREEN_FLAGS),
            "blue": sorted(result.flags & BLUE_FLAGS),
            "unparsed_chunks": result.unparsed_chunks,
            "total_chunks": result.total_chunks,
        }
