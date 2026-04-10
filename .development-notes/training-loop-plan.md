# Training Loop Implementation Plan

Plan for checklist items 8 (LoRA Fine-Tuning) and related testing. Covers everything from
environment setup through merged model export. Refers to final_adr.md for architectural
decisions and collated-flag-set.md for the flag taxonomy.

## Decisions Made

- Training data: use data/train.jsonl as-is (RAG context already baked into examples). ChromaDB
  setup deferred to inference pipeline.
- Environment: SageMaker Space (JupyterLab), ml.g6.xlarge (1x L4, 24GB VRAM).
- Framework: trl.SFTTrainer + peft for LoRA. Minimizes custom code vs HW6's manual loop.
- Chat template: on-the-fly conversion via tokenizer.apply_chat_template (not pre-converted).
  Overhead is negligible compared to GPU forward/backward passes.
- Model: meta-llama/Llama-3.2-3B-Instruct in float16.
- Testing: 80% coverage minimum, model/Bedrock calls mocked.

## LoRA Configuration (Starting Point)

Borrowed from HW6, adjusted for 3B model. Tune from eval results.

- r (rank): 16
- alpha: 32
- dropout: 0.05
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- bias: all
- task_type: CAUSAL_LM
- use_rslora: True

## Training Hyperparameters (Starting Point)

- epochs: 3 (HW6 did 2; our task is narrower, watch eval loss for overfitting)
- batch_size: 4
- gradient_accumulation_steps: 4 (effective batch size 16)
- learning_rate: 2e-5
- warmup_ratio: 0.1
- weight_decay: 0.01
- max_grad_norm: 1.0
- max_seq_length: TBD (step 2 below will determine; likely 1024-2048)
- eval_strategy: epoch
- save_strategy: epoch
- fp16: True

## Steps

### 1. Write the plan

Write this file. You're reading it.

### 2. Check token lengths in training data

Before setting max_seq_length, measure the actual token distribution in data/train.jsonl after
chat template expansion. Script in scripts/check_token_lengths.py. Outputs min, max, mean, p95,
p99 token counts. This tells us whether 1024 is sufficient or if we need 2048.

### 3. Add training dependencies to pyproject.toml

Add a training dependency group (not main deps — these are SageMaker-only):

- torch
- transformers
- peft
- trl
- datasets
- accelerate

### 4. Write the training script

scripts/train.py — the core deliverable. Structure:

a. Load tokenizer and set pad_token = eos_token
b. Load data/train.jsonl and data/eval.jsonl as datasets
c. Format examples on-the-fly: extract messages list, apply chat template
d. Configure LoRA via peft (config above)
e. Load base model in float16 with device_map=auto
f. Apply LoRA via get_peft_model
g. Configure SFTTrainer with eval dataset, logging, checkpointing
h. Train
i. Save adapter checkpoint

SFTTrainer handles tokenization, label masking, and the training loop. The script is primarily
configuration and data loading glue.

### 5. Write the merge/export script

scripts/merge_and_export.py — post-training step required for Bedrock Custom Model Import.

a. Load base model
b. Load LoRA adapter from checkpoint
c. Merge adapter into base model (model.merge_and_unload())
d. Save merged model as HF safetensors
e. Save tokenizer alongside (Bedrock needs it)
f. Fix tokenizer_class field in tokenizer_config.json if needed (known Bedrock quirk from HW6:
   must be "LlamaTokenizerFast", not the backend class name)

### 6. Write tests

tests/test_training.py — covers the training script's components:

a. Data loading: verify JSONL parsing, message structure, field presence
b. Chat template formatting: verify messages convert to expected Llama template with special tokens
c. LoRA config: verify config values, target modules
d. Merge/export: mock model loading, verify merge_and_unload is called, verify save paths
e. Token length script: verify statistics computation on sample data

All model loading and .from_pretrained calls mocked. Tests run on CPU without GPU or model
weights. Target 80%+ coverage on training code.

### 7. Validate on SageMaker

Not code — manual validation steps once scripts are written:

a. Spin up ml.g6.xlarge SageMaker Space
b. Clone repo, install training deps (uv sync --group training)
c. Run token length check, confirm max_seq_length setting
d. Run training script, monitor loss curves
e. Evaluate on eval set (precision, recall, format compliance)
f. If results are acceptable, run merge/export
g. Verify merged model loads and generates reasonable output

### 8. Update checklist

Mark checklist items complete as each step finishes.

## File Map

- scripts/check_token_lengths.py (step 2)
- scripts/train.py (step 4)
- scripts/merge_and_export.py (step 5)
- tests/test_training.py (step 6)

## Notes

- SageMaker quirk: venvs must be on local disk, not S3-mounted home dir.
- Llama 3.2 3B in float16 is ~6GB. With LoRA adapters, optimizer states, and batch activations,
  expect ~16-20GB peak VRAM on the L4. Should fit.
- If 3B OOMs during training, reduce batch_size to 2 and increase gradient_accumulation_steps to
  8 (same effective batch size).
- eval.jsonl is used during training for loss/metric monitoring. test.jsonl (202 records) is
  reserved for final evaluation after training — never seen during training.
