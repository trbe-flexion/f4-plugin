# Retrain Plan: Realistic RAG Context

## Problem

The model was trained with `simulate_rag_context(flag, flag_seeds)` — the RAG context in every
training example used seed passages from the *correct* flag. The model learned to treat RAG context
as a strong hint about what flag to look for.

At inference, the real ChromaDB retriever queries by *chunk similarity*, returning passages about
topically similar RFPs — not necessarily the same flag. This mismatch causes degeneration (gibberish
output with RAG, format non-compliance without RAG).

## Root Cause

`generate_data.py` line 340:
```python
flag_examples.append(make_example(chunk, simulate_rag_context(flag, flag_seeds), flag))
```

`simulate_rag_context` randomly samples from `seeds[flag]` — always the correct flag's seeds.

## Fix

Replace `simulate_rag_context` with actual ChromaDB retrieval during data generation. The training
data should reflect what the model will see at inference: noisy context that may or may not contain
the correct flag's examples.

## Steps

### 1. Update data generation script

Modify `generate_data.py` to:

a. Pre-populate a ChromaDB store from `data/rag_seeds.jsonl` (all flags, same as inference)
b. For each generated chunk, query the store by chunk text (top_k=3) — same as `FlagRAGStore.query`
c. Format context using `format_context()` from `src/rag/retriever.py` — same as inference
d. This means training examples will sometimes get relevant context, sometimes irrelevant context,
   sometimes mixed — matching real-world behavior

Key: the chunk text is generated first, then we retrieve against it. The retrieval may or may not
return passages for the correct flag. The model must learn to detect flags from the chunk itself,
using RAG context as helpful-but-unreliable supplementary information.

### 2. Regenerate training data

Run the updated `generate_data.py`. This produces new `train.jsonl` and `eval.jsonl` with
realistic RAG context. The chunks themselves don't change (or can be regenerated) — only the
RAG context portion of the user message changes.

Option: regenerate only the RAG context for existing chunks (cheaper, no Claude API calls needed)
by re-wrapping existing chunk text with real retrieval results. This avoids regenerating the
chunks themselves.

### 3. Retrain on SageMaker

Same process as before:
```bash
uv run python training/train.py --max-seq-length 2048
```

### 4. Re-evaluate

```bash
PYTHONPATH=. uv run python training/evaluate.py --compare
```

Expect: format compliance should stay high. Precision/recall may shift — the model is now trained
on harder examples where context doesn't hand it the answer.

### 5. Re-merge and re-deploy

```bash
uv run python training/merge_and_export.py
aws s3 sync models/merged/ s3://trbe-f4-finetuned-model/ --region us-east-1
```

Then re-import to Bedrock (new import job, new model ARN).

### 6. Re-test with real RFP

Run the debug pipeline and Gradio demo against the VA solicitation again.

## Also Fix: Parser Leniency

Independent of retraining. The model sometimes outputs `none`, `None`, or comma-separated flags.
Update `parse_flags` to handle:

- `none` / `None` → treat as `no_flag`
- Comma-separated on one line → split by comma as well as newline
- Strip surrounding prose if a valid flag name appears within a line

## Questions

- Do we regenerate chunks from scratch (full Claude API cost) or reuse existing chunks and only
  re-wrap the RAG context? Re-wrapping is much cheaper and isolates the variable we're changing.
- Should we also add training examples *without* RAG context? The model may encounter situations
  where RAG retrieval returns nothing. Currently every training example has context.
- Keep the same hyperparameters or adjust? Probably keep the same as a baseline, adjust if
  eval results warrant it.
