from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from src.chunking.chunker import chunk_text
from src.decision.engine import FilterDecisionEngine
from src.domain.entities import FilterResult
from src.domain.parsing import parse_flags
from src.domain.protocols import FlagDetector

logger = logging.getLogger(__name__)


class F4Pipeline:
    """Orchestrates the F4 filtering pipeline.

    Chunks text, runs inference concurrently, parses output,
    deduplicates flags, and applies decision logic.
    """

    def __init__(
        self,
        flag_detector: FlagDetector,
        tokenizer,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        max_workers: int = 4,
        red_flag_threshold: int = 999,
    ):
        self.flag_detector = flag_detector
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.max_workers = max_workers
        self.decision_engine = FilterDecisionEngine(red_flag_threshold)

    def _process_chunk(self, chunk: str) -> list[str] | None:
        """Detect flags in a single chunk with one retry on parse failure.

        Returns list of flags (possibly empty for no_flag), or None if
        both attempts produced unparseable output.
        """
        for _attempt in range(2):
            raw_output = self.flag_detector.detect_flags(chunk)
            flags = parse_flags(raw_output)
            if flags or raw_output.strip() == "no_flag":
                return flags
        return None

    def filter(self, text: str) -> FilterResult:
        """Run the full F4 pipeline on RFP text."""
        chunks = chunk_text(text, self.max_tokens, self.overlap_tokens, self.tokenizer)

        if not chunks:
            return FilterResult(filter=False)

        all_flags: set[str] = set()
        unparsed_chunks = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
            for future in futures:
                try:
                    chunk_flags = future.result()
                    if chunk_flags is None:
                        unparsed_chunks += 1
                    elif chunk_flags:
                        all_flags.update(chunk_flags)
                except Exception:
                    unparsed_chunks += 1
                    logger.exception("Error processing chunk")

        logger.info(
            "F4 pipeline complete: %d chunks, %d flags, %d unparsed",
            len(chunks),
            len(all_flags),
            unparsed_chunks,
        )
        if all_flags:
            logger.info("Detected flags: %s", sorted(all_flags))

        decision = self.decision_engine.decide(all_flags)
        return FilterResult(filter=decision)
