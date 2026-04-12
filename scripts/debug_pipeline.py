"""Debug script to see raw model output per chunk.

Usage:
    uv run python scripts/debug_pipeline.py --model-arn <ARN> --file <path>
"""

from __future__ import annotations

import argparse

from src.chunking.chunker import chunk_text
from src.domain.parsing import parse_flags
from src.inference.bedrock import BedrockFlagDetector
from src.rag.retriever import format_context
from src.rag.store import FlagRAGStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arn", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--no-rag", action="store_true")
    args = parser.parse_args()

    from src.frontend.extraction import extract_text

    text = extract_text(args.file)
    print(f"Extracted: {len(text)} chars, ~{len(text.split())} words\n")

    chunks = chunk_text(text, max_tokens=512, overlap_tokens=64)
    print(f"Chunks: {len(chunks)}\n")

    detector = BedrockFlagDetector(model_id=args.model_arn, region=args.region)

    rag_store = None if args.no_rag else FlagRAGStore.get_or_init()

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1}/{len(chunks)} ({len(chunk.split())} words) ---")
        print(f"Text preview: {chunk[:100]}...")
        print()

        if rag_store:
            results = rag_store.query(chunk, top_k=3)
            if results:
                prompt = format_context(results, chunk)
                print(f"RAG context added ({len(results)} results)")
            else:
                prompt = chunk
                print("No RAG results")
        else:
            prompt = chunk

        raw = detector.detect_flags(prompt)
        flags = parse_flags(raw)
        is_no_flag = raw.strip() == "no_flag"

        print(f"Raw output: {repr(raw)}")
        print(f"Parsed flags: {flags}")
        print(f"Would count as unparsed: {not flags and not is_no_flag}")
        print()


if __name__ == "__main__":
    main()
