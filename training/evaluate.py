"""Evaluate fine-tuned model on held-out test set.

Supports three modes: base model only, fine-tuned only, or both with
a side-by-side comparison table.

Designed to run on SageMaker after training completes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.domain.parsing import parse_flags

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_test_examples(path: str) -> list[dict]:
    """Load test examples, extracting input text and ground truth flags."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record["messages"]

            assistant_content = next(m["content"] for m in messages if m["role"] == "assistant")

            ground_truth = set(parse_flags(assistant_content))
            if not ground_truth and assistant_content.strip() == "no_flag":
                ground_truth = set()

            examples.append(
                {
                    "messages": [m for m in messages if m["role"] in ("system", "user")],
                    "ground_truth": ground_truth,
                    "raw_ground_truth": assistant_content.strip(),
                }
            )
    return examples


def compute_metrics(results: list[dict]) -> dict:
    """Compute precision, recall, and format compliance from results."""
    total_predicted = 0
    total_correct_predicted = 0
    total_ground_truth = 0
    total_correct_recalled = 0
    total_chunks = len(results)
    parseable_chunks = 0

    for r in results:
        predicted = r["predicted"]
        truth = r["ground_truth"]

        if r["format_ok"]:
            parseable_chunks += 1

        total_predicted += len(predicted)
        total_ground_truth += len(truth)

        correct = predicted & truth
        total_correct_predicted += len(correct)
        total_correct_recalled += len(correct)

    precision = total_correct_predicted / total_predicted if total_predicted > 0 else 1.0
    recall = total_correct_recalled / total_ground_truth if total_ground_truth > 0 else 1.0
    compliance = parseable_chunks / total_chunks if total_chunks > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0,
        "format_compliance": round(compliance, 4),
        "total_chunks": total_chunks,
        "parseable_chunks": parseable_chunks,
        "total_predicted_flags": total_predicted,
        "total_ground_truth_flags": total_ground_truth,
        "total_correct": total_correct_predicted,
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    """Print metrics in a readable format."""
    if label:
        print(f"\n{label}")
        print("=" * 40)
    print(f"Chunks:            {metrics['total_chunks']}")
    print(f"Format compliance: {metrics['format_compliance']:.1%}")
    print(f"Precision:         {metrics['precision']:.1%}")
    print(f"Recall:            {metrics['recall']:.1%}")
    print(f"F1:                {metrics['f1']:.1%}")
    print(f"Predicted flags:   {metrics['total_predicted_flags']}")
    print(f"Ground truth flags:{metrics['total_ground_truth_flags']}")
    print(f"Correct:           {metrics['total_correct']}")


def print_per_flag_metrics(results: list[dict]) -> None:
    """Print per-flag precision and recall."""
    flag_tp: dict[str, int] = {}
    flag_fp: dict[str, int] = {}
    flag_fn: dict[str, int] = {}

    for r in results:
        predicted = r["predicted"]
        truth = r["ground_truth"]

        for flag in predicted:
            if flag in truth:
                flag_tp[flag] = flag_tp.get(flag, 0) + 1
            else:
                flag_fp[flag] = flag_fp.get(flag, 0) + 1

        for flag in truth:
            if flag not in predicted:
                flag_fn[flag] = flag_fn.get(flag, 0) + 1

    all_flags = sorted(set(flag_tp) | set(flag_fp) | set(flag_fn))

    print(f"\n{'Flag':<30} {'Prec':>6} {'Rec':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 60)
    for flag in all_flags:
        tp = flag_tp.get(flag, 0)
        fp = flag_fp.get(flag, 0)
        fn = flag_fn.get(flag, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"{flag:<30} {prec:>5.1%} {rec:>5.1%} {tp:>4} {fp:>4} {fn:>4}")


def print_comparison(base_metrics: dict, finetuned_metrics: dict) -> None:
    """Print side-by-side comparison of base vs fine-tuned metrics."""
    print("\n" + "=" * 60)
    print("COMPARISON: Base Model vs Fine-Tuned")
    print("=" * 60)

    rows = [
        ("Format compliance", "format_compliance", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1", "f1", True),
    ]

    print(f"{'Metric':<22} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 56)
    for label, key, is_pct in rows:
        base_val = base_metrics[key]
        ft_val = finetuned_metrics[key]
        delta = ft_val - base_val
        sign = "+" if delta > 0 else ""
        if is_pct:
            print(f"{label:<22} {base_val:>9.1%} {ft_val:>11.1%} {sign}{delta:>8.1%}")
        else:
            print(f"{label:<22} {base_val:>10} {ft_val:>12} {sign}{delta:>10}")


def run_inference(model, tokenizer, examples: list[dict]) -> list[dict]:
    """Run inference on examples and return results with predictions."""
    import torch

    results = []
    for i, example in enumerate(examples):
        prompt = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predicted = set(parse_flags(raw_output))

        format_ok = bool(predicted) or raw_output.strip() == "no_flag"

        results.append(
            {
                "predicted": predicted,
                "ground_truth": example["ground_truth"],
                "raw_output": raw_output.strip(),
                "raw_ground_truth": example["raw_ground_truth"],
                "format_ok": format_ok,
            }
        )

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(examples)}")

    return results


def save_results(metrics: dict, results: list[dict], path: str) -> None:
    """Save metrics and results to JSON."""
    serializable = []
    for r in results:
        serializable.append(
            {
                "predicted": sorted(r["predicted"]),
                "ground_truth": sorted(r["ground_truth"]),
                "raw_output": r["raw_output"],
                "raw_ground_truth": r["raw_ground_truth"],
                "format_ok": r["format_ok"],
            }
        )
    output_data = {"metrics": metrics, "results": serializable}
    with open(path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {path}")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on test set")
    parser.add_argument("--test-data", type=str, default="data/test.jsonl")
    parser.add_argument("--adapter-dir", type=str, default="models/adapter")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--output-dir", type=str, default="evaluation")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--base-only", action="store_true", help="Evaluate base model only")
    mode.add_argument(
        "--finetuned-only",
        action="store_true",
        help="Evaluate fine-tuned model only (default)",
    )
    mode.add_argument("--compare", action="store_true", help="Evaluate both and print comparison")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parsed = parse_args(args)

    test_path = Path(parsed.test_data)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    run_base = parsed.base_only or parsed.compare
    run_finetuned = (
        parsed.finetuned_only or parsed.compare or (not parsed.base_only and not parsed.compare)
    )

    if run_finetuned:
        adapter_path = Path(parsed.adapter_dir)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {parsed.model}")
    tokenizer = AutoTokenizer.from_pretrained(parsed.model)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading test data: {test_path}")
    examples = load_test_examples(str(test_path))
    print(f"  {len(examples)} examples")

    base_metrics = None
    finetuned_metrics = None

    if run_base:
        print(f"\nLoading base model: {parsed.model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            parsed.model, dtype=torch.float16, device_map="auto"
        )
        base_model.eval()

        print("Running base model inference...")
        base_results = run_inference(base_model, tokenizer, examples)
        base_metrics = compute_metrics(base_results)
        print_metrics(base_metrics, label="Base Model")
        print_per_flag_metrics(base_results)
        save_results(base_metrics, base_results, str(output_dir / "baseline.json"))

        # Free memory before loading fine-tuned model
        if run_finetuned:
            del base_model
            torch.cuda.empty_cache()

    if run_finetuned:
        print(f"\nLoading model: {parsed.model}")
        model = AutoModelForCausalLM.from_pretrained(
            parsed.model, dtype=torch.float16, device_map="auto"
        )
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

        print("Running fine-tuned model inference...")
        ft_results = run_inference(model, tokenizer, examples)
        finetuned_metrics = compute_metrics(ft_results)
        print_metrics(finetuned_metrics, label="Fine-Tuned Model")
        print_per_flag_metrics(ft_results)
        save_results(finetuned_metrics, ft_results, str(output_dir / "finetuned.json"))

    if parsed.compare and base_metrics and finetuned_metrics:
        print_comparison(base_metrics, finetuned_metrics)


if __name__ == "__main__":
    main()
