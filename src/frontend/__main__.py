"""Entry point for the F4 Gradio demo.

Usage:
    uv run python -m src.frontend --model-arn <ARN> --share --auth user:password
"""

from __future__ import annotations

import argparse

from src.frontend.app import launch_app
from src.inference.bedrock import BedrockFlagDetector
from src.pipeline.filter import F4Pipeline
from src.rag.store import FlagRAGStore


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F4 Gradio demo")
    parser.add_argument("--model-arn", required=True, help="Bedrock imported model ARN")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--share", action="store_true", help="Create public Gradio tunnel")
    parser.add_argument("--auth", help="user:password for Gradio auth")
    parser.add_argument("--max-tokens", type=int, default=512, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=64, help="Chunk overlap in tokens")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent Bedrock calls")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)

    detector = BedrockFlagDetector(
        model_id=parsed.model_arn,
        region=parsed.region,
    )

    rag_store = None if parsed.no_rag else FlagRAGStore.get_or_init()

    pipeline = F4Pipeline(
        flag_detector=detector,
        max_tokens=parsed.max_tokens,
        overlap_tokens=parsed.overlap,
        max_workers=parsed.max_workers,
        rag_store=rag_store,
    )

    auth = None
    if parsed.auth and ":" in parsed.auth:
        user, password = parsed.auth.split(":", 1)
        auth = (user, password)

    launch_app(pipeline, auth=auth, share=parsed.share)


if __name__ == "__main__":
    main()
