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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, help="Write full prompts and outputs to file")
    args = parser.parse_args()

    from src.frontend.extraction import extract_text

    text = extract_text(args.file)
    print(f"Extracted: {len(text)} chars, ~{len(text.split())} words\n")

    chunks = chunk_text(text, max_tokens=args.max_tokens, overlap_tokens=64)
    print(f"Chunks: {len(chunks)}\n")

    detector = BedrockFlagDetector(model_id=args.model_arn, region=args.region)

    rag_store = None if args.no_rag else FlagRAGStore.get_or_init()

    results_log = []

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1}/{len(chunks)} ({len(chunk.split())} words) ---")
        print(f"Text preview: {chunk[:100]}...")
        print()

        if rag_store:
            rag_results = rag_store.query(chunk, top_k=args.top_k)
            if rag_results:
                prompt = format_context(rag_results, chunk)
                print(f"RAG context added ({len(rag_results)} results)")
            else:
                prompt = chunk
                print("No RAG results")
        else:
            prompt = chunk

        raw = detector.detect_flags(prompt)
        flags = parse_flags(raw)
        is_no_flag = raw.strip().lower() in ("no_flag", "none")

        print(f"Raw output: {repr(raw)}")
        print(f"Parsed flags: {flags}")
        print(f"Would count as unparsed: {not flags and not is_no_flag}")
        print()

        results_log.append((i, len(chunks), prompt, raw, flags))

    if args.output:
        with open(args.output, "w") as out_file:
            for i, total, prompt, raw, flags in results_log:
                out_file.write(f"{'=' * 80}\n")
                out_file.write(f"CHUNK {i + 1}/{total}\n")
                out_file.write(f"{'=' * 80}\n\n")
                out_file.write("--- FULL PROMPT (sent to model as user message) ---\n\n")
                out_file.write(prompt)
                out_file.write("\n\n--- MODEL OUTPUT ---\n\n")
                out_file.write(raw)
                out_file.write("\n\n--- PARSED ---\n")
                out_file.write(f"Flags: {list(flags)}\n\n")
        print(f"Full prompts/outputs written to {args.output}")


if __name__ == "__main__":
    main()
