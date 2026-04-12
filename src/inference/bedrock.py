"""Bedrock FlagDetector adapter.

Implements the FlagDetector protocol by calling a fine-tuned model
deployed via Bedrock Custom Model Import.
"""

from __future__ import annotations

import json

import boto3

# Must match the system prompt used during training (see .development-notes/system-prompt.md)
SYSTEM_PROMPT = """\
You are a flag detection model for government RFP screening. \
Your job is to identify which flags, if any, are present in the provided RFP text chunk.

You will be given:
- A list of relevant flag definitions and examples (retrieved context)
- An RFP text chunk to analyze

Rules:
- Output one flag name per line, using only the exact flag names listed below
- If no flags are present, output: no_flag
- Do not output explanations, reasoning, or any other text
- Only detect a flag when the evidence in the chunk is explicit — do not infer from weak signals

Valid flags:
waterfall_methodology, off_the_shelf_software, no_custom_development, lpta_source_selection,
small_business_set_aside, 8a_set_aside, wosb_set_aside, edwosb_set_aside, sdvosb_set_aside,
hubzone_set_aside, agile_methodology, oral_presentation, design_exercise, budget_too_low,
brownfield, onsite_required, onsite_madison, large_team, marginal_short_duration, no_flag"""


class BedrockFlagDetector:
    """FlagDetector backed by a Bedrock Custom Model Import endpoint."""

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        max_gen_len: int = 64,
        temperature: float = 0.1,
    ):
        self.model_id = model_id
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def _format_prompt(self, chunk: str) -> str:
        """Wrap chunk in Llama 3.2 Instruct chat template."""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{chunk}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def detect_flags(self, chunk: str) -> str:
        """Send chunk to Bedrock and return raw model output."""
        prompt = self._format_prompt(chunk)
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": self.max_gen_len,
                "temperature": self.temperature,
            }
        )
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        result = json.loads(response["body"].read())
        return result.get("generation", "")
