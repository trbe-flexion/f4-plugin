from unittest.mock import MagicMock

from src.domain.entities import FilterResult
from src.pipeline.filter import F4Pipeline


def make_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: list(range(len(text)))
    tokenizer.decode.side_effect = lambda ids: "x" * len(ids)
    return tokenizer


def make_mock_detector(responses):
    """Create a mock FlagDetector that returns responses in order."""
    detector = MagicMock()
    detector.detect_flags.side_effect = list(responses)
    return detector


class TestF4Pipeline:
    def test_empty_text_no_filter(self):
        detector = make_mock_detector([])
        tokenizer = make_mock_tokenizer()
        pipeline = F4Pipeline(detector, tokenizer)

        result = pipeline.filter("")

        assert result == FilterResult(filter=False)
        detector.detect_flags.assert_not_called()

    def test_single_chunk_black_flag(self):
        detector = make_mock_detector(["waterfall_methodology"])
        tokenizer = make_mock_tokenizer()
        pipeline = F4Pipeline(detector, tokenizer, max_tokens=1000)

        result = pipeline.filter("short text")

        assert result.filter is True

    def test_single_chunk_no_flag(self):
        detector = make_mock_detector(["no_flag"])
        tokenizer = make_mock_tokenizer()
        pipeline = F4Pipeline(detector, tokenizer, max_tokens=1000)

        result = pipeline.filter("short text")

        assert result.filter is False

    def test_multiple_chunks_with_flags(self):
        detector = make_mock_detector(
            [
                "brownfield",
                "no_flag",
                "waterfall_methodology",
            ]
        )
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(30))
        tokenizer.decode.side_effect = lambda ids: "x" * len(ids)

        pipeline = F4Pipeline(detector, tokenizer, max_tokens=10, overlap_tokens=0)
        result = pipeline.filter("a" * 30)

        assert result.filter is True  # waterfall is black

    def test_deduplicates_flags(self):
        detector = make_mock_detector(
            [
                "brownfield",
                "brownfield",
            ]
        )
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(20))
        tokenizer.decode.side_effect = lambda ids: "x" * len(ids)

        pipeline = F4Pipeline(detector, tokenizer, max_tokens=10, overlap_tokens=0)
        result = pipeline.filter("a" * 20)

        assert result.filter is False  # brownfield is red, default threshold=999

    def test_retry_on_parse_failure(self):
        detector = MagicMock()
        detector.detect_flags.side_effect = [
            "garbage output",
            "waterfall_methodology",
        ]
        tokenizer = make_mock_tokenizer()
        pipeline = F4Pipeline(detector, tokenizer, max_tokens=1000)

        result = pipeline.filter("some text")

        assert result.filter is True
        assert detector.detect_flags.call_count == 2

    def test_unparsed_after_both_attempts(self):
        detector = MagicMock()
        detector.detect_flags.return_value = "total garbage"
        tokenizer = make_mock_tokenizer()
        pipeline = F4Pipeline(detector, tokenizer, max_tokens=1000)

        result = pipeline.filter("some text")

        assert result.filter is False
        assert detector.detect_flags.call_count == 2

    def test_red_flag_threshold(self):
        detector = make_mock_detector(
            [
                "brownfield\nlpta_source_selection",
            ]
        )
        tokenizer = make_mock_tokenizer()
        pipeline = F4Pipeline(detector, tokenizer, max_tokens=1000, red_flag_threshold=2)

        result = pipeline.filter("some text")

        assert result.filter is True

    def test_green_flags_pass(self):
        detector = make_mock_detector(["agile_methodology\noral_presentation"])
        tokenizer = make_mock_tokenizer()
        pipeline = F4Pipeline(detector, tokenizer, max_tokens=1000)

        result = pipeline.filter("some text")

        assert result.filter is False

    def test_exception_in_detector(self):
        detector = MagicMock()
        detector.detect_flags.side_effect = RuntimeError("model error")
        tokenizer = make_mock_tokenizer()
        pipeline = F4Pipeline(detector, tokenizer, max_tokens=1000)

        result = pipeline.filter("some text")

        assert result.filter is False  # graceful degradation
