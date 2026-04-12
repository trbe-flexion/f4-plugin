"""Merge LoRA adapter back into base model and export for Bedrock.

After fine-tuning, this script merges the LoRA adapter weights into the
base Llama 3.2 3B model and saves the result as HuggingFace safetensors.
The output is ready for Bedrock Custom Model Import.

Known Bedrock quirk: tokenizer_class in tokenizer_config.json must be
the user-facing class name (e.g. "LlamaTokenizerFast"), not the backend
class. This script fixes it automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EXPECTED_TOKENIZER_CLASS = "LlamaTokenizerFast"


def fix_tokenizer_class(output_dir: str) -> bool:
    """Fix tokenizer_class in tokenizer_config.json for Bedrock compatibility.

    Returns True if a fix was applied, False if already correct or file missing.
    """
    config_path = Path(output_dir) / "tokenizer_config.json"
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = json.load(f)

    current = config.get("tokenizer_class")
    if current == EXPECTED_TOKENIZER_CLASS:
        return False

    config["tokenizer_class"] = EXPECTED_TOKENIZER_CLASS
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Fixed tokenizer_class: {current!r} -> {EXPECTED_TOKENIZER_CLASS!r}")
    return True


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export for Bedrock")
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="models/adapter",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/merged",
        help="Path to save merged model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help="Base model ID",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)

    adapter_path = Path(parsed.adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(parsed.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {parsed.model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        parsed.model,
        dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(str(output_path), safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(parsed.model)
    tokenizer.save_pretrained(str(output_path))

    fix_tokenizer_class(str(output_path))

    print("Done. Output ready for Bedrock Custom Model Import.")


if __name__ == "__main__":
    main()
