from unittest.mock import MagicMock

from src.chunking.chunker import chunk_text


def make_mock_tokenizer(token_per_char=True):
    """Create a mock tokenizer where each character is one token."""
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: list(range(len(text)))
    tokenizer.decode.side_effect = lambda ids: "x" * len(ids)
    return tokenizer


class TestChunkText:
    def test_short_text_single_chunk(self):
        tokenizer = make_mock_tokenizer()
        chunks = chunk_text("hello", max_tokens=10, overlap_tokens=2, tokenizer=tokenizer)
        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_empty_text(self):
        tokenizer = make_mock_tokenizer()
        assert chunk_text("", max_tokens=10, overlap_tokens=2, tokenizer=tokenizer) == []

    def test_whitespace_only(self):
        tokenizer = make_mock_tokenizer()
        assert chunk_text("   ", max_tokens=10, overlap_tokens=2, tokenizer=tokenizer) == []

    def test_exact_fit(self):
        tokenizer = make_mock_tokenizer()
        text = "a" * 10
        chunks = chunk_text(text, max_tokens=10, overlap_tokens=2, tokenizer=tokenizer)
        assert len(chunks) == 1

    def test_splits_long_text(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(20))
        tokenizer.decode.side_effect = lambda ids: "x" * len(ids)

        chunks = chunk_text("a" * 20, max_tokens=10, overlap_tokens=2, tokenizer=tokenizer)
        assert len(chunks) == 3  # 0-10, 8-18, 16-20

    def test_overlap_present(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(20))
        tokenizer.decode.side_effect = lambda ids: f"chunk({ids[0]}-{ids[-1]})"

        chunks = chunk_text("a" * 20, max_tokens=10, overlap_tokens=3, tokenizer=tokenizer)

        # With max=10, overlap=3, step=7: chunks at 0-10, 7-17, 14-20
        assert len(chunks) == 3

    def test_no_overlap(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(20))
        tokenizer.decode.side_effect = lambda ids: "x" * len(ids)

        chunks = chunk_text("a" * 20, max_tokens=10, overlap_tokens=0, tokenizer=tokenizer)
        assert len(chunks) == 2  # 0-10, 10-20
