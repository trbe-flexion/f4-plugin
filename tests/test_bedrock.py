"""Tests for BedrockFlagDetector adapter.

All boto3 calls are mocked. Tests run without AWS credentials or live endpoints.
"""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from src.inference.bedrock import SYSTEM_PROMPT, BedrockFlagDetector


@pytest.fixture
def mock_boto3_client():
    with patch("src.inference.bedrock.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        yield mock_client


def make_bedrock_response(generation: str) -> dict:
    """Build a mock Bedrock invoke_model response."""
    body = json.dumps(
        {
            "generation": generation,
            "prompt_token_count": 200,
            "generation_token_count": 5,
            "stop_reason": "stop",
        }
    )
    return {"body": BytesIO(body.encode())}


class TestBedrockFlagDetectorInit:
    def test_stores_model_id(self, mock_boto3_client):
        detector = BedrockFlagDetector(model_id="arn:aws:bedrock:us-east-1:123:imported-model/abc")
        assert detector.model_id == "arn:aws:bedrock:us-east-1:123:imported-model/abc"

    def test_default_region(self, mock_boto3_client):
        with patch("src.inference.bedrock.boto3") as mock_boto3:
            BedrockFlagDetector(model_id="arn:test")
            mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")

    def test_custom_region(self, mock_boto3_client):
        with patch("src.inference.bedrock.boto3") as mock_boto3:
            BedrockFlagDetector(model_id="arn:test", region="us-west-2")
            mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

    def test_default_parameters(self, mock_boto3_client):
        detector = BedrockFlagDetector(model_id="arn:test")
        assert detector.max_gen_len == 64
        assert detector.temperature == 0.1

    def test_custom_parameters(self, mock_boto3_client):
        detector = BedrockFlagDetector(model_id="arn:test", max_gen_len=128, temperature=0.5)
        assert detector.max_gen_len == 128
        assert detector.temperature == 0.5


class TestFormatPrompt:
    def test_contains_system_prompt(self, mock_boto3_client):
        detector = BedrockFlagDetector(model_id="arn:test")
        prompt = detector._format_prompt("Some RFP text.")
        assert SYSTEM_PROMPT in prompt

    def test_contains_chunk(self, mock_boto3_client):
        detector = BedrockFlagDetector(model_id="arn:test")
        prompt = detector._format_prompt("The contractor shall use waterfall.")
        assert "The contractor shall use waterfall." in prompt

    def test_has_chat_template_structure(self, mock_boto3_client):
        detector = BedrockFlagDetector(model_id="arn:test")
        prompt = detector._format_prompt("chunk text")
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
        assert prompt.endswith("\n\n")

    def test_system_before_user(self, mock_boto3_client):
        detector = BedrockFlagDetector(model_id="arn:test")
        prompt = detector._format_prompt("chunk")
        sys_pos = prompt.index("system")
        user_pos = prompt.index("user")
        asst_pos = prompt.index("assistant")
        assert sys_pos < user_pos < asst_pos


class TestDetectFlags:
    def test_returns_generation(self, mock_boto3_client):
        mock_boto3_client.invoke_model.return_value = make_bedrock_response("waterfall_methodology")
        detector = BedrockFlagDetector(model_id="arn:test")
        result = detector.detect_flags("Use waterfall approach.")
        assert result == "waterfall_methodology"

    def test_returns_no_flag(self, mock_boto3_client):
        mock_boto3_client.invoke_model.return_value = make_bedrock_response("no_flag")
        detector = BedrockFlagDetector(model_id="arn:test")
        result = detector.detect_flags("Generic IT services.")
        assert result == "no_flag"

    def test_returns_multiple_flags(self, mock_boto3_client):
        mock_boto3_client.invoke_model.return_value = make_bedrock_response(
            "off_the_shelf_software\nno_custom_development"
        )
        detector = BedrockFlagDetector(model_id="arn:test")
        result = detector.detect_flags("COTS only, no custom dev.")
        assert "off_the_shelf_software" in result
        assert "no_custom_development" in result

    def test_passes_model_id(self, mock_boto3_client):
        mock_boto3_client.invoke_model.return_value = make_bedrock_response("no_flag")
        detector = BedrockFlagDetector(model_id="arn:aws:bedrock:us-east-1:123:imported-model/xyz")
        detector.detect_flags("text")
        call_kwargs = mock_boto3_client.invoke_model.call_args[1]
        assert call_kwargs["modelId"] == "arn:aws:bedrock:us-east-1:123:imported-model/xyz"

    def test_request_body_format(self, mock_boto3_client):
        mock_boto3_client.invoke_model.return_value = make_bedrock_response("no_flag")
        detector = BedrockFlagDetector(model_id="arn:test", max_gen_len=32, temperature=0.2)
        detector.detect_flags("chunk text")
        call_kwargs = mock_boto3_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        assert body["max_gen_len"] == 32
        assert body["temperature"] == 0.2
        assert "prompt" in body

    def test_empty_generation(self, mock_boto3_client):
        mock_boto3_client.invoke_model.return_value = make_bedrock_response("")
        detector = BedrockFlagDetector(model_id="arn:test")
        result = detector.detect_flags("text")
        assert result == ""


class TestSystemPrompt:
    def test_contains_all_valid_flags(self):
        from src.domain.taxonomy import VALID_FLAGS

        for flag in VALID_FLAGS:
            assert flag in SYSTEM_PROMPT, f"Missing flag: {flag}"

    def test_contains_no_flag(self):
        assert "no_flag" in SYSTEM_PROMPT

    def test_contains_rules(self):
        assert "one flag name per line" in SYSTEM_PROMPT
        assert "do not infer from weak signals" in SYSTEM_PROMPT


class TestProtocolCompliance:
    def test_implements_flag_detector(self, mock_boto3_client):
        from src.domain.protocols import FlagDetector

        detector = BedrockFlagDetector(model_id="arn:test")
        assert isinstance(detector, FlagDetector)
