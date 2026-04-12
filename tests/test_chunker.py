from unittest.mock import MagicMock

from src.chunking.chunker import TOKENS_PER_WORD, chunk_text


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


class TestChunkTextWordBased:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("hello world", max_tokens=512, overlap_tokens=64)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_empty_text(self):
        assert chunk_text("", max_tokens=512, overlap_tokens=64) == []

    def test_whitespace_only(self):
        assert chunk_text("   ", max_tokens=512, overlap_tokens=64) == []

    def test_splits_long_text(self):
        # 1.3 tokens/word, max_tokens=10 -> ~7 words per chunk
        words = ["word"] * 20
        text = " ".join(words)
        chunks = chunk_text(text, max_tokens=10, overlap_tokens=2)
        assert len(chunks) > 1
        for chunk in chunks:
            chunk_words = chunk.split()
            assert len(chunk_words) <= int(10 / TOKENS_PER_WORD) + 1

    def test_overlap_present(self):
        words = [f"w{i}" for i in range(30)]
        text = " ".join(words)
        chunks = chunk_text(text, max_tokens=13, overlap_tokens=4)
        # With overlap, chunks should share words at boundaries
        assert len(chunks) > 1
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        assert first_words & second_words  # non-empty overlap

    def test_no_tokenizer_needed(self):
        chunks = chunk_text("some text here", max_tokens=512, overlap_tokens=64)
        assert len(chunks) == 1
        assert chunks[0] == "some text here"

    def test_preserves_all_words(self):
        words = [f"w{i}" for i in range(15)]
        text = " ".join(words)
        chunks = chunk_text(text, max_tokens=10, overlap_tokens=2)
        all_chunk_words = set()
        for chunk in chunks:
            all_chunk_words.update(chunk.split())
        assert all_chunk_words == set(words)
