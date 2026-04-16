"""Tests for training scripts.

All model loading and HF calls are mocked. Tests run on CPU
without GPU or model weights.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from training.check_token_lengths import (
    compute_statistics,
    compute_token_lengths,
    load_messages,
    print_statistics,
)
from training.merge_and_export import (
    EXPECTED_TOKENIZER_CLASS,
    fix_tokenizer_class,
)
from training.merge_and_export import (
    parse_args as merge_parse_args,
)
from training.train import (
    load_jsonl_records,
    make_formatting_func,
)
from training.train import (
    parse_args as train_parse_args,
)

# --- Fixtures ---


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file with chat-format messages."""
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are a flag detection model."},
                {"role": "user", "content": "Analyze this RFP chunk."},
                {"role": "assistant", "content": "no_flag"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a flag detection model."},
                {"role": "user", "content": "This RFP requires waterfall."},
                {"role": "assistant", "content": "waterfall_methodology"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a flag detection model."},
                {"role": "user", "content": "COTS solution only, no development."},
                {
                    "role": "assistant",
                    "content": "off_the_shelf_software\nno_custom_development",
                },
            ]
        },
    ]
    jsonl_path = tmp_path / "test_data.jsonl"
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return jsonl_path


@pytest.fixture
def empty_jsonl(tmp_path):
    """Create an empty JSONL file."""
    jsonl_path = tmp_path / "empty.jsonl"
    jsonl_path.write_text("")
    return jsonl_path


# --- check_token_lengths tests ---


class TestLoadMessages:
    def test_loads_messages_from_jsonl(self, sample_jsonl):
        messages_list = load_messages(str(sample_jsonl))
        assert len(messages_list) == 3
        assert messages_list[0][0]["role"] == "system"
        assert messages_list[1][2]["content"] == "waterfall_methodology"

    def test_skips_blank_lines(self, tmp_path):
        jsonl_path = tmp_path / "with_blanks.jsonl"
        record = {"messages": [{"role": "user", "content": "hello"}]}
        jsonl_path.write_text(f"\n{json.dumps(record)}\n\n{json.dumps(record)}\n")
        messages_list = load_messages(str(jsonl_path))
        assert len(messages_list) == 2

    def test_empty_file(self, empty_jsonl):
        messages_list = load_messages(str(empty_jsonl))
        assert messages_list == []


class TestComputeTokenLengths:
    def test_returns_lengths(self, sample_jsonl):
        messages_list = load_messages(str(sample_jsonl))
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "templated text"
        mock_tokenizer.encode.return_value = list(range(50))

        lengths = compute_token_lengths(messages_list, mock_tokenizer)

        assert len(lengths) == 3
        assert all(length == 50 for length in lengths)
        assert mock_tokenizer.apply_chat_template.call_count == 3

    def test_varying_lengths(self):
        messages_list = [
            [{"role": "user", "content": "short"}],
            [{"role": "user", "content": "longer message here"}],
        ]
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"
        mock_tokenizer.encode.side_effect = [list(range(10)), list(range(100))]

        lengths = compute_token_lengths(messages_list, mock_tokenizer)

        assert lengths == [10, 100]


class TestComputeStatistics:
    def test_basic_statistics(self):
        lengths = [100, 200, 300, 400, 500]
        stats = compute_statistics(lengths)
        assert stats["count"] == 5
        assert stats["min"] == 100
        assert stats["max"] == 500
        assert stats["mean"] == 300.0
        assert stats["median"] == 300

    def test_percentiles(self):
        lengths = list(range(1, 101))
        stats = compute_statistics(lengths)
        assert stats["p95"] == 96
        assert stats["p99"] == 100

    def test_single_element(self):
        stats = compute_statistics([42])
        assert stats["count"] == 1
        assert stats["min"] == 42
        assert stats["max"] == 42
        assert stats["mean"] == 42.0
        assert stats["p95"] == 42
        assert stats["p99"] == 42


class TestPrintStatistics:
    def test_prints_all_fields(self, capsys):
        stats = {
            "count": 10,
            "min": 50,
            "max": 500,
            "mean": 275.0,
            "median": 250.0,
            "p95": 480,
            "p99": 500,
        }
        print_statistics(stats)
        output = capsys.readouterr().out
        assert "Examples: 10" in output
        assert "Min:      50" in output
        assert "Max:      500" in output
        assert "P95:      480" in output
        assert "P99:      500" in output


class TestCheckTokenLengthsMain:
    def test_main_with_sample_data(self, sample_jsonl, capsys):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"
        mock_tokenizer.encode.return_value = list(range(200))

        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_transformers = MagicMock(AutoTokenizer=mock_auto_tokenizer)
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            from training.check_token_lengths import main

            main(["--data", str(sample_jsonl)])

        output = capsys.readouterr().out
        assert "Examples: 3" in output
        assert "Recommendation: max_seq_length=1024" in output

    def test_main_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            from training.check_token_lengths import main

            main(["--data", str(tmp_path / "nonexistent.jsonl")])


# --- train.py tests ---


class TestLoadJsonlRecords:
    def test_loads_records(self, sample_jsonl):
        records = load_jsonl_records(str(sample_jsonl))
        assert len(records) == 3
        assert "messages" in records[0]

    def test_message_structure(self, sample_jsonl):
        records = load_jsonl_records(str(sample_jsonl))
        first = records[0]["messages"]
        assert len(first) == 3
        assert first[0]["role"] == "system"
        assert first[1]["role"] == "user"
        assert first[2]["role"] == "assistant"

    def test_skips_blank_lines(self, tmp_path):
        jsonl_path = tmp_path / "blanks.jsonl"
        record = {"messages": [{"role": "user", "content": "hi"}]}
        jsonl_path.write_text(f"\n{json.dumps(record)}\n\n")
        records = load_jsonl_records(str(jsonl_path))
        assert len(records) == 1

    def test_empty_file(self, empty_jsonl):
        records = load_jsonl_records(str(empty_jsonl))
        assert records == []


class TestMakeFormattingFunc:
    def test_applies_chat_template(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<|begin|>formatted<|end|>"

        func = make_formatting_func(mock_tokenizer)
        example = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }
        result = func(example)

        assert result == "<|begin|>formatted<|end|>"
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            example["messages"], tokenize=False, add_generation_prompt=False
        )


class TestBuildLoraConfig:
    def test_default_config(self):
        mock_lora_config = MagicMock()
        mock_task_type = MagicMock()
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"
        mock_peft = MagicMock()
        mock_peft.LoraConfig = mock_lora_config
        mock_peft.TaskType = mock_task_type

        with patch.dict(sys.modules, {"peft": mock_peft}):
            from training.train import build_lora_config

            build_lora_config()

        mock_lora_config.assert_called_once()
        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["r"] == 16
        assert call_kwargs["lora_alpha"] == 32
        assert call_kwargs["lora_dropout"] == 0.05
        assert call_kwargs["use_rslora"] is True
        assert "q_proj" in call_kwargs["target_modules"]
        assert "down_proj" in call_kwargs["target_modules"]

    def test_override_rank(self):
        mock_lora_config = MagicMock()
        mock_task_type = MagicMock()
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"
        mock_peft = MagicMock()
        mock_peft.LoraConfig = mock_lora_config
        mock_peft.TaskType = mock_task_type

        with patch.dict(sys.modules, {"peft": mock_peft}):
            from training.train import build_lora_config

            build_lora_config(r=8)

        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["r"] == 8
        assert call_kwargs["lora_alpha"] == 32


class TestTrainParseArgs:
    def test_defaults(self):
        args = train_parse_args([])
        assert args.train_data == "data/train.jsonl"
        assert args.eval_data == "data/eval.jsonl"
        assert args.output_dir == "models/adapter"
        assert args.epochs == 3
        assert args.batch_size == 4
        assert args.gradient_accumulation_steps == 4
        assert args.learning_rate == 2e-5
        assert args.max_seq_length == 1024

    def test_overrides(self):
        args = train_parse_args(
            [
                "--epochs",
                "5",
                "--batch-size",
                "2",
                "--max-seq-length",
                "2048",
                "--lora-r",
                "8",
            ]
        )
        assert args.epochs == 5
        assert args.batch_size == 2
        assert args.max_seq_length == 2048
        assert args.lora_r == 8


# --- merge_and_export.py tests ---


class TestFixTokenizerClass:
    def test_fixes_wrong_class(self, tmp_path):
        config = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4096}
        config_path = tmp_path / "tokenizer_config.json"
        config_path.write_text(json.dumps(config))

        result = fix_tokenizer_class(str(tmp_path))

        assert result is True
        updated = json.loads(config_path.read_text())
        assert updated["tokenizer_class"] == EXPECTED_TOKENIZER_CLASS
        assert updated["model_max_length"] == 4096

    def test_already_correct(self, tmp_path):
        config = {"tokenizer_class": EXPECTED_TOKENIZER_CLASS}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))

        result = fix_tokenizer_class(str(tmp_path))

        assert result is False

    def test_missing_file(self, tmp_path):
        result = fix_tokenizer_class(str(tmp_path))
        assert result is False

    def test_missing_key(self, tmp_path):
        config = {"model_max_length": 4096}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))

        result = fix_tokenizer_class(str(tmp_path))

        assert result is True
        updated = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert updated["tokenizer_class"] == EXPECTED_TOKENIZER_CLASS


class TestMergeParseArgs:
    def test_defaults(self):
        args = merge_parse_args([])
        assert args.adapter_dir == "models/adapter"
        assert args.output_dir == "models/merged"
        assert args.model == "meta-llama/Llama-3.2-3B-Instruct"

    def test_overrides(self):
        args = merge_parse_args(
            [
                "--adapter-dir",
                "custom/adapter",
                "--output-dir",
                "custom/merged",
            ]
        )
        assert args.adapter_dir == "custom/adapter"
        assert args.output_dir == "custom/merged"


class TestMergeMain:
    def test_merge_flow(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = tmp_path / "merged"

        mock_base = MagicMock()
        mock_merged = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_base
        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model
        mock_tokenizer_cls = MagicMock()
        mock_torch = MagicMock()

        mock_modules = {
            "torch": mock_torch,
            "peft": MagicMock(PeftModel=mock_peft_cls),
            "transformers": MagicMock(
                AutoModelForCausalLM=mock_auto_model,
                AutoTokenizer=mock_tokenizer_cls,
            ),
        }

        with patch.dict(sys.modules, mock_modules):
            from training.merge_and_export import main

            with (
                patch("training.merge_and_export.fix_tokenizer_class") as mock_fix_tok,
                patch("training.merge_and_export.fix_config_for_bedrock") as mock_fix_cfg,
            ):
                main(
                    [
                        "--adapter-dir",
                        str(adapter_dir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

        mock_auto_model.from_pretrained.assert_called_once()
        mock_peft_cls.from_pretrained.assert_called_once_with(mock_base, str(adapter_dir))
        mock_peft_model.merge_and_unload.assert_called_once()
        mock_merged.save_pretrained.assert_called_once_with(
            str(output_dir), safe_serialization=True
        )
        mock_fix_tok.assert_called_once_with(str(output_dir))
        mock_fix_cfg.assert_called_once_with(str(output_dir))

    def test_missing_adapter_dir(self, tmp_path):
        from training.merge_and_export import main

        with pytest.raises(FileNotFoundError, match="Adapter directory not found"):
            main(["--adapter-dir", str(tmp_path / "nonexistent")])
