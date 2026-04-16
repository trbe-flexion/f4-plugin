# Handoff: Run 8 — Demo Model

## Current State

Run 8 is the demo model. Trained on 7 flags (reduced from 14), evaluated on
held-out test set: **88.7% F1, 92.2% precision, 85.5% recall**. Adapter is
on SageMaker at `models/adapter/`. Training and eval are complete.

## What Was Done This Session

1. **Cost analysis killed the relabeling approach** — full Opus relabeling of
   958 RFPs estimated at $1,400–$2,300. Too expensive.

2. **Pivoted to flag reduction** — instead of relabeling, dropped the 7
   worst-performing flags from Run 6 results and rebuilt the training set
   from existing `opus_validated_real.jsonl` data.

3. **Kept 7 flags**: oral_presentation, small_business_set_aside,
   agile_methodology, lpta_source_selection, 8a_set_aside, sdvosb_set_aside,
   hubzone_set_aside.

4. **Dropped 7 flags**: design_exercise (0/0), large_team (0/0),
   marginal_short_duration (50/10), budget_too_low (100/25),
   onsite_required (56/83 FP-heavy), wosb_set_aside (57/100 FP-heavy),
   off_the_shelf_software (67/33).

5. **Code changes**:
   - `scripts/build_training_set.py`: Added `KEEP_FLAGS` constant that filters
     non-kept flags during Step 0 (strips flags from records, records with no
     remaining flags become no_flag).
   - `scripts/system-prompt.md`: Updated to list only 7 flags + no_flag.
   - Old data files moved to `data/archive/`.

6. **Training**: 5 epochs, 80/10/10 split, batch 8 / grad accum 2,
   negative ratio 0.2, no RAG, max_seq_length 2048. Same config as Run 6
   except the split (was 90/10/0).

## Next Steps: Merge + Deploy to Bedrock

### 1. Merge LoRA adapter back to base model

On SageMaker:

```bash
PYTHONPATH=. uv run python scripts/merge_adapter.py
```

Check that this script exists and outputs to the expected path (likely
`models/merged/`). If it doesn't exist, the merge is straightforward:
load base model + PeftModel, call `merge_and_unload()`, save.

### 2. Export as HF safetensors

The merged model should already be in safetensors format from the merge step.
Verify the output directory contains `model.safetensors` (or sharded
`model-00001-of-*.safetensors`) + `config.json` + `tokenizer.json`.

### 3. Upload to S3 for Bedrock Custom Model Import

```bash
aws s3 cp models/merged/ s3://<bucket>/f4-model/ --recursive
```

### 4. Bedrock Custom Model Import

Terraform config is in the repo. Create the custom model import job pointing
at the S3 path. Once imported, note the model ARN for the inference client.

### 5. Update inference client

Point the Bedrock inference client at the new model ARN. Test with the
`scripts/test_bedrock_live.py` script.

### 6. Demo prep

The Gradio frontend (`src/frontend.py`) needs to be wired to the Bedrock
endpoint. Upload a real RFP, show flag detection results. The 7-flag model
should produce clean, high-confidence results suitable for a live demo.

## Key Files

| File | Role |
|------|------|
| `models/adapter/` | LoRA adapter (on SageMaker) |
| `data/train.jsonl` | Training data (7 flags, 80% split) |
| `data/eval.jsonl` | Eval split (10%) |
| `data/test.jsonl` | Held-out test split (10%) |
| `data/archive/opus_validated_real.jsonl` | Source data (all flags) |
| `scripts/build_training_set.py` | Build script with KEEP_FLAGS filter |
| `scripts/system-prompt.md` | System prompt (7 flags) |
| `evaluation/evaluate.py` | Eval script |
| `evaluation/finetuned.json` | Run 8 detailed results |

## Open Questions

- **wosb_set_aside**: Dropped due to FP-heavy results (57/100 in Run 6), but
  it had 42 examples in the source data. Could potentially be recovered with
  cleaner training data if needed post-demo.
- **No-flag hallucination rate**: 37.5% (3/8) on test set. Small sample, but
  worth watching. The model occasionally predicts flags on clean chunks.
- **Bedrock context limit**: Likely 8192 tokens for Llama 3 family. Not
  relevant for current 2048 training but could enable larger chunks in future.
