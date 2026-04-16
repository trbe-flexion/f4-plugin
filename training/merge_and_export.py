"""Merge LoRA adapter back into base model and export for Bedrock.

After fine-tuning, this script merges the LoRA adapter weights into the
base Llama 3.2 3B model and saves the result as HuggingFace safetensors.
The output is ready for Bedrock Custom Model Import.

Known Bedrock quirks fixed automatically:
- tokenizer_class must be "LlamaTokenizerFast" (not backend class name)
- config.json must use transformers 4.51.3 field names (rope_scaling not
  rope_parameters), factor=8.0, top-level rope_theta, version=4.51.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EXPECTED_TOKENIZER_CLASS = "LlamaTokenizerFast"


def fix_config_for_bedrock(output_dir: str) -> bool:
    """Fix config.json for Bedrock compatibility (expects transformers 4.51.3 format).

    Applies all known fixes:
    - rope_parameters -> rope_scaling (5.x -> 4.x field name)
    - rope_scaling.factor -> 8.0 (Bedrock's documented override for Llama 3)
    - rope_theta moved to top level (4.x format)
    - transformers_version -> 4.51.3

    Without these, long prompts (>~875 tokens) degenerate into gibberish on Bedrock.
    See .development-notes/notes/bedrock-deployment.md section 6 for full investigation.

    Returns True if any fix was applied, False if already correct or file missing.
    """
    config_path = Path(output_dir) / "config.json"
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = json.load(f)

    changed = False

    # rope_parameters -> rope_scaling (transformers 5.x -> 4.x)
    if "rope_parameters" in config:
        config["rope_scaling"] = config.pop("rope_parameters")
        print("Fixed config.json: rope_parameters -> rope_scaling")
        changed = True

    # Set rope_scaling.factor to 8.0
    if "rope_scaling" in config:
        if config["rope_scaling"].get("factor") != 8.0:
            config["rope_scaling"]["factor"] = 8.0
            print("Fixed config.json: rope_scaling.factor -> 8.0")
            changed = True

        # Move rope_theta to top level if nested inside rope_scaling
        if "rope_theta" in config["rope_scaling"]:
            config["rope_theta"] = config["rope_scaling"].pop("rope_theta")
            print("Fixed config.json: moved rope_theta to top level")
            changed = True

    # Ensure rope_theta is set at top level
    if "rope_theta" not in config:
        config["rope_theta"] = 500000.0
        print("Fixed config.json: set rope_theta = 500000.0")
        changed = True

    # Downgrade transformers_version to match Bedrock's inference container
    if config.get("transformers_version") != "4.51.3":
        old_ver = config.get("transformers_version", "(missing)")
        config["transformers_version"] = "4.51.3"
        print(f"Fixed config.json: transformers_version {old_ver} -> 4.51.3")
        changed = True

    if changed:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

    return changed


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
    fix_config_for_bedrock(str(output_path))

    print("Done. Output ready for Bedrock Custom Model Import.")


if __name__ == "__main__":
    main()
