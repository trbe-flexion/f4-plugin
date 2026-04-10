"""LoRA fine-tuning script for F4 flag detection model.

Uses SFTTrainer from trl with peft LoRA on Llama 3.2 3B Instruct.
Training data is OpenAI chat-format JSONL with RAG context baked in.
Chat template conversion happens on-the-fly via the tokenizer.

Designed to run on SageMaker ml.g6.xlarge (L4 24GB VRAM).
"""

from __future__ import annotations

import argparse
import json

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_jsonl_records(path: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_formatting_func(tokenizer):
    """Return a formatting function that applies the chat template to each example."""

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    return formatting_func


def build_lora_config(**overrides):
    """Build a LoRA config with defaults, allowing overrides."""
    from peft import LoraConfig, TaskType

    defaults = {
        "r": DEFAULT_LORA_RANK,
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "lora_dropout": DEFAULT_LORA_DROPOUT,
        "target_modules": DEFAULT_LORA_TARGET_MODULES,
        "bias": "all",
        "task_type": TaskType.CAUSAL_LM,
        "use_rslora": True,
    }
    config = {**defaults, **overrides}
    return LoraConfig(**config)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B for F4 flag detection")
    parser.add_argument("--train-data", type=str, default="data/train.jsonl")
    parser.add_argument("--eval-data", type=str, default="data/eval.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/adapter")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    parsed = parse_args(args)

    print(f"Loading tokenizer: {parsed.model}")
    tokenizer = AutoTokenizer.from_pretrained(parsed.model)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    train_dataset = Dataset.from_list(load_jsonl_records(parsed.train_data))
    eval_dataset = Dataset.from_list(load_jsonl_records(parsed.eval_data))
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval:  {len(eval_dataset)} examples")

    print(f"Loading model: {parsed.model}")
    model = AutoModelForCausalLM.from_pretrained(
        parsed.model,
        dtype=torch.float16,
        device_map="auto",
    )

    lora_config = build_lora_config(r=parsed.lora_r, lora_alpha=parsed.lora_alpha)

    total_steps = (
        len(train_dataset)
        // (parsed.batch_size * parsed.gradient_accumulation_steps)
        * parsed.epochs
    )
    warmup_steps = int(total_steps * parsed.warmup_ratio)

    sft_config = SFTConfig(
        output_dir=parsed.output_dir,
        num_train_epochs=parsed.epochs,
        per_device_train_batch_size=parsed.batch_size,
        per_device_eval_batch_size=parsed.batch_size,
        gradient_accumulation_steps=parsed.gradient_accumulation_steps,
        learning_rate=parsed.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=parsed.weight_decay,
        max_grad_norm=parsed.max_grad_norm,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=parsed.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {parsed.output_dir}")
    trainer.save_model(parsed.output_dir)
    tokenizer.save_pretrained(parsed.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
