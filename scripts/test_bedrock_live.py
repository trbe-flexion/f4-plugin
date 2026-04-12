"""Manual smoke test for Bedrock Custom Model Import endpoint.

NOT run in CI — requires live AWS credentials and a deployed model.

Usage:
    python scripts/test_bedrock_live.py --model-arn <ARN>
    python scripts/test_bedrock_live.py --model-arn <ARN> --region us-east-1
"""

from __future__ import annotations

import argparse
import json

import boto3

SYSTEM_PROMPT = (
    "You are a flag detection model for government RFP screening. "
    "Your job is to identify which flags, if any, are present in the provided RFP text chunk.\n\n"
    "Rules:\n"
    "- Output one flag name per line, using only the exact flag names listed below\n"
    "- If no flags are present, output: no_flag\n"
    "- Do not output explanations, reasoning, or any other text\n"
    "- Only detect a flag when the evidence in the chunk is explicit\n\n"
    "Valid flags:\n"
    "waterfall_methodology, off_the_shelf_software, no_custom_development, lpta_source_selection, "
    "small_business_set_aside, 8a_set_aside, wosb_set_aside, edwosb_set_aside, sdvosb_set_aside, "
    "hubzone_set_aside, agile_methodology, oral_presentation, design_exercise, budget_too_low, "
    "brownfield, onsite_required, onsite_madison, large_team, marginal_short_duration, no_flag"
)

TEST_CASES = [
    {
        "name": "Waterfall (should detect)",
        "text": (
            "The contractor shall follow a traditional waterfall software development lifecycle."
        ),
        "expected": "waterfall_methodology",
    },
    {
        "name": "Agile (should detect)",
        "text": (
            "We require an agile development approach with two-week sprints and daily standups."
        ),
        "expected": "agile_methodology",
    },
    {
        "name": "No flags (should be clean)",
        "text": "The government seeks a contractor to provide IT modernization services.",
        "expected": "no_flag",
    },
    {
        "name": "COTS (should detect)",
        "text": (
            "The solution must be a commercially available"
            " off-the-shelf product with no custom development."
        ),
        "expected": "off_the_shelf_software",
    },
    {
        "name": "Small business set-aside",
        "text": "This procurement is set aside exclusively for small business concerns.",
        "expected": "small_business_set_aside",
    },
]


def format_prompt(system: str, user: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def invoke(client, model_arn: str, user_text: str) -> dict:
    prompt = format_prompt(SYSTEM_PROMPT, user_text)
    body = json.dumps(
        {
            "prompt": prompt,
            "max_gen_len": 64,
            "temperature": 0.1,
        }
    )
    response = client.invoke_model(modelId=model_arn, body=body)
    return json.loads(response["body"].read())


def main():
    parser = argparse.ArgumentParser(description="Smoke test Bedrock endpoint")
    parser.add_argument("--model-arn", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    client = boto3.client("bedrock-runtime", region_name=args.region)

    print(f"Model: {args.model_arn}")
    print(f"Region: {args.region}")
    print("=" * 60)

    passed = 0
    for tc in TEST_CASES:
        result = invoke(client, args.model_arn, tc["text"])
        generation = result.get("generation", "").strip()
        match = tc["expected"] in generation
        status = "PASS" if match else "FAIL"
        if match:
            passed += 1

        print(f"\n[{status}] {tc['name']}")
        print(f"  Expected: {tc['expected']}")
        print(f"  Got:      {generation}")
        prompt_tokens = result.get("prompt_token_count", "?")
        gen_tokens = result.get("generation_token_count", "?")
        print(f"  Tokens:   {prompt_tokens} in, {gen_tokens} out")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()
