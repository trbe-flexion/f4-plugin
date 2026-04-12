from __future__ import annotations

TOKENS_PER_WORD = 1.3


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer=None,
) -> list[str]:
    """Split text into overlapping chunks.

    If tokenizer is provided (must support encode/decode), uses exact token
    boundaries. Otherwise uses word-based approximation (~1.3 tokens/word).
    """
    if not text or not text.strip():
        return []

    if tokenizer is not None:
        return _chunk_by_tokens(text, max_tokens, overlap_tokens, tokenizer)
    return _chunk_by_words(text, max_tokens, overlap_tokens)


def _chunk_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer,
) -> list[str]:
    """Split text at exact token boundaries."""
    token_ids = tokenizer.encode(text)

    if len(token_ids) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    step = max_tokens - overlap_tokens

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids))

        if end == len(token_ids):
            break
        start += step

    return chunks


def _chunk_by_words(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text by words, approximating token counts."""
    words = text.split()
    max_words = int(max_tokens / TOKENS_PER_WORD)
    overlap_words = int(overlap_tokens / TOKENS_PER_WORD)

    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    step = max_words - overlap_words

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break
        start += step

    return chunks
