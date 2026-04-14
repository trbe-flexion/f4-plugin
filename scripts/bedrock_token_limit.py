"""Binary search for Bedrock's effective prompt token limit.

Sends progressively longer prompts to find where the model degenerates.
Uses a simple repeated-word prompt to control length precisely.

Usage:
    PYTHONPATH=. uv run python scripts/bedrock_token_limit.py --model-arn <ARN>
    PYTHONPATH=. uv run python scripts/bedrock_token_limit.py --model-arn <ARN> --format messages
"""

from __future__ import annotations

import argparse
import json

import boto3

from src.inference.bedrock import SYSTEM_PROMPT


def format_prompt_raw(system: str, user: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


WATERFALL_SUFFIX = (
    " The contractor shall follow a traditional waterfall software development lifecycle."
)


def invoke_raw(client, model_id: str, n_words: int) -> dict:
    padding = " ".join(["government"] * n_words)
    prompt = format_prompt_raw(SYSTEM_PROMPT, padding + WATERFALL_SUFFIX)
    body = json.dumps(
        {
            "prompt": prompt,
            "max_gen_len": 32,
            "temperature": 0.1,
        }
    )
    response = client.invoke_model(modelId=model_id, body=body)
    return json.loads(response["body"].read())


def invoke_messages(client, model_id: str, n_words: int) -> dict:
    padding = " ".join(["government"] * n_words)
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": padding + WATERFALL_SUFFIX},
            ],
            "max_tokens": 32,
            "temperature": 0.1,
        }
    )
    response = client.invoke_model(modelId=model_id, body=body)
    return json.loads(response["body"].read())


def get_token_count(result: dict, fmt: str) -> int:
    if fmt == "messages":
        return result.get("usage", {}).get("prompt_tokens", 0)
    return result.get("prompt_token_count", 0)


def get_generation(result: dict, fmt: str) -> str:
    if fmt == "messages":
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
    return result.get("generation", "")


def test_length(client, model_id: str, n_words: int, fmt: str) -> tuple[int, bool]:
    if fmt == "messages":
        result = invoke_messages(client, model_id, n_words)
    else:
        result = invoke_raw(client, model_id, n_words)

    token_count = get_token_count(result, fmt)
    generation = get_generation(result, fmt).strip()
    passed = "waterfall_methodology" in generation
    return token_count, passed


def main():
    parser = argparse.ArgumentParser(description="Binary search for Bedrock prompt token limit")
    parser.add_argument("--model-arn", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--format",
        choices=["raw", "messages"],
        default="raw",
        help="API format to test",
    )
    args = parser.parse_args()

    client = boto3.client("bedrock-runtime", region_name=args.region)
    fmt = args.format

    print(f"Format: {fmt}")
    print("Phase 1: Find approximate range")
    print("=" * 60)

    lo_words = 0
    hi_words = 50
    last_good_tokens = 0

    while hi_words < 5000:
        tokens, passed = test_length(client, args.model_arn, hi_words, fmt)
        status = "PASS" if passed else "FAIL"
        print(f"  {hi_words:>5} padding words, {tokens:>5} tokens: {status}")
        if not passed:
            break
        last_good_tokens = tokens
        lo_words = hi_words
        hi_words *= 2
    else:
        print("No failure found up to 5000 words.")
        return

    print(f"\nFailure between {lo_words} and {hi_words} padding words")
    print(f"Last good token count: {last_good_tokens}")

    print("\nPhase 2: Binary search")
    print("=" * 60)

    for _ in range(10):
        if hi_words - lo_words <= 5:
            break
        mid = (lo_words + hi_words) // 2
        tokens, passed = test_length(client, args.model_arn, mid, fmt)
        status = "PASS" if passed else "FAIL"
        print(f"  {mid:>5} padding words, {tokens:>5} tokens: {status}")
        if passed:
            lo_words = mid
            last_good_tokens = tokens
        else:
            hi_words = mid

    tokens_lo, _ = test_length(client, args.model_arn, lo_words, fmt)
    tokens_hi, _ = test_length(client, args.model_arn, hi_words, fmt)

    print(f"\nResult: model degenerates between {tokens_lo} and {tokens_hi} prompt tokens")
    print(f"Safe limit: ~{tokens_lo} tokens")


if __name__ == "__main__":
    main()
