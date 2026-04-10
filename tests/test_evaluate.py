"""Tests for evaluation script."""

import json

import pytest

from evaluation.evaluate import (
    compute_metrics,
    load_test_examples,
    parse_args,
    print_comparison,
    print_metrics,
    print_per_flag_metrics,
    save_results,
)


@pytest.fixture
def sample_test_jsonl(tmp_path):
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are a flag detection model."},
                {"role": "user", "content": "Analyze this chunk."},
                {"role": "assistant", "content": "waterfall_methodology"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a flag detection model."},
                {"role": "user", "content": "Another chunk."},
                {"role": "assistant", "content": "no_flag"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a flag detection model."},
                {"role": "user", "content": "Multi-flag chunk."},
                {"role": "assistant", "content": "brownfield\nlpta_source_selection"},
            ]
        },
    ]
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


class TestLoadTestExamples:
    def test_loads_examples(self, sample_test_jsonl):
        examples = load_test_examples(sample_test_jsonl)
        assert len(examples) == 3

    def test_single_flag_ground_truth(self, sample_test_jsonl):
        examples = load_test_examples(sample_test_jsonl)
        assert examples[0]["ground_truth"] == {"waterfall_methodology"}

    def test_no_flag_ground_truth(self, sample_test_jsonl):
        examples = load_test_examples(sample_test_jsonl)
        assert examples[1]["ground_truth"] == set()

    def test_multi_flag_ground_truth(self, sample_test_jsonl):
        examples = load_test_examples(sample_test_jsonl)
        assert examples[2]["ground_truth"] == {"brownfield", "lpta_source_selection"}

    def test_messages_exclude_assistant(self, sample_test_jsonl):
        examples = load_test_examples(sample_test_jsonl)
        for ex in examples:
            roles = [m["role"] for m in ex["messages"]]
            assert "assistant" not in roles


class TestComputeMetrics:
    def test_perfect_predictions(self):
        results = [
            {
                "predicted": {"waterfall_methodology"},
                "ground_truth": {"waterfall_methodology"},
                "format_ok": True,
            },
            {"predicted": set(), "ground_truth": set(), "format_ok": True},
        ]
        metrics = compute_metrics(results)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["format_compliance"] == 1.0

    def test_all_wrong(self):
        results = [
            {
                "predicted": {"brownfield"},
                "ground_truth": {"waterfall_methodology"},
                "format_ok": True,
            },
        ]
        metrics = compute_metrics(results)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_partial_match(self):
        results = [
            {
                "predicted": {"brownfield", "waterfall_methodology"},
                "ground_truth": {"brownfield", "lpta_source_selection"},
                "format_ok": True,
            },
        ]
        metrics = compute_metrics(results)
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5

    def test_format_compliance(self):
        results = [
            {"predicted": set(), "ground_truth": set(), "format_ok": True},
            {"predicted": set(), "ground_truth": {"brownfield"}, "format_ok": False},
            {"predicted": set(), "ground_truth": set(), "format_ok": True},
        ]
        metrics = compute_metrics(results)
        assert abs(metrics["format_compliance"] - 0.6667) < 0.001

    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics["total_chunks"] == 0
        assert metrics["format_compliance"] == 0.0


class TestPrintMetrics:
    def test_prints_output(self, capsys):
        metrics = {
            "total_chunks": 10,
            "format_compliance": 0.9,
            "precision": 0.85,
            "recall": 0.8,
            "f1": 0.8235,
            "total_predicted_flags": 20,
            "total_ground_truth_flags": 18,
            "total_correct": 15,
        }
        print_metrics(metrics)
        output = capsys.readouterr().out
        assert "Chunks:" in output
        assert "Precision:" in output
        assert "Recall:" in output

    def test_prints_label(self, capsys):
        metrics = {
            "total_chunks": 5,
            "format_compliance": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "total_predicted_flags": 5,
            "total_ground_truth_flags": 5,
            "total_correct": 5,
        }
        print_metrics(metrics, label="Base Model")
        output = capsys.readouterr().out
        assert "Base Model" in output


class TestPrintPerFlagMetrics:
    def test_prints_per_flag(self, capsys):
        results = [
            {
                "predicted": {"waterfall_methodology"},
                "ground_truth": {"waterfall_methodology"},
            },
            {"predicted": {"brownfield"}, "ground_truth": set()},
        ]
        print_per_flag_metrics(results)
        output = capsys.readouterr().out
        assert "waterfall_methodology" in output
        assert "brownfield" in output


class TestPrintComparison:
    def test_prints_comparison_table(self, capsys):
        base = {
            "precision": 0.5,
            "recall": 0.4,
            "f1": 0.4444,
            "format_compliance": 0.7,
        }
        finetuned = {
            "precision": 0.9,
            "recall": 0.85,
            "f1": 0.8743,
            "format_compliance": 0.95,
        }
        print_comparison(base, finetuned)
        output = capsys.readouterr().out
        assert "COMPARISON" in output
        assert "Base" in output
        assert "Fine-tuned" in output
        assert "Delta" in output
        assert "Precision" in output

    def test_shows_positive_delta(self, capsys):
        base = {"precision": 0.5, "recall": 0.5, "f1": 0.5, "format_compliance": 0.5}
        finetuned = {"precision": 0.9, "recall": 0.9, "f1": 0.9, "format_compliance": 0.9}
        print_comparison(base, finetuned)
        output = capsys.readouterr().out
        assert "+" in output


class TestSaveResults:
    def test_saves_json(self, tmp_path):
        metrics = {"precision": 0.9, "recall": 0.8}
        results = [
            {
                "predicted": {"brownfield"},
                "ground_truth": {"brownfield"},
                "raw_output": "brownfield",
                "raw_ground_truth": "brownfield",
                "format_ok": True,
            }
        ]
        path = str(tmp_path / "results.json")
        save_results(metrics, results, path)

        with open(path) as f:
            data = json.load(f)
        assert data["metrics"]["precision"] == 0.9
        assert len(data["results"]) == 1
        assert data["results"][0]["predicted"] == ["brownfield"]


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.test_data == "data/test.jsonl"
        assert args.adapter_dir == "models/adapter"
        assert not args.base_only
        assert not args.compare

    def test_base_only(self):
        args = parse_args(["--base-only"])
        assert args.base_only is True
        assert not args.compare

    def test_compare(self):
        args = parse_args(["--compare"])
        assert args.compare is True
        assert not args.base_only

    def test_finetuned_only(self):
        args = parse_args(["--finetuned-only"])
        assert args.finetuned_only is True

    def test_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            parse_args(["--base-only", "--compare"])
