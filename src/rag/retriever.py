"""Format retrieved RAG passages into the context block the model expects."""

from __future__ import annotations


def format_context(results: list[dict], chunk: str) -> str:
    """Format retrieved passages and chunk into the prompt format from training data."""
    parts = ["[Retrieved context]"]

    for i, result in enumerate(results, 1):
        parts.append(f"\nExample {i}:\n{result['passage']}")

    parts.append(f"\n---\n\nRFP chunk to analyze:\n{chunk}")

    return "\n".join(parts)
