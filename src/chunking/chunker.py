from __future__ import annotations


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer,
) -> list[str]:
    """Split text into overlapping chunks at token boundaries.

    tokenizer must support encode(str) -> list[int] and decode(list[int]) -> str.
    """
    if not text or not text.strip():
        return []

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
