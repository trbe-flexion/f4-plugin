<h1 align="center">
  <img src="assets/f4_logo-1.png" alt="F4 Logo" width="64">&nbsp;
  F4: Flexion Fast Fail Filtering
  &nbsp;<img src="assets/f4_logo-1.png" alt="F4 Logo" width="64">
</h1>

A Python library that screens government RFP text for disqualifying flags before expensive LLM analysis. F4 uses a fine-tuned Llama 3.2 3B model deployed to AWS Bedrock, delivering chunk-level flag detection at a fraction of the cost of a large model.

Designed as a standalone plug-in for the [opp-capture](https://github.com/trbe-flexion/flexion-opp-capture) pipeline. If F4 disappears, opp-capture keeps working.

## Running the Demo

Requires AWS credentials with Bedrock InvokeModel permission (Alt/cohort account).

```bash
uv sync
uv run python -m src.frontend \
  --model-arn "arn:aws:bedrock:us-east-1:165286508758:imported-model/pxi20ybyyh5t" \
  --share --auth demo:demo
```

Upload a PDF or DOCX. F4 chunks the text, runs inference against the Bedrock endpoint, and displays detected flags with a FILTER/REVIEW recommendation. First invocation after idle hits a cold start (~1-2 min).

Options: `--max-workers N` (concurrent Bedrock calls), `--region REGION`.

## How It Works

```
f4.filter(text)
  -> chunk text into ~512-token windows with overlap
  -> send chunks concurrently to Bedrock (bounded ThreadPoolExecutor)
  -> parse model output (one flag name per line)
  -> retry once on parse failure, discard on second failure
  -> deduplicate flags across all chunks
  -> apply algorithmic decision logic
  -> return FilterResult
```

The model outputs one flag name per line, nothing else. The application layer handles taxonomy lookup, tier assignment, and decision logic. Format compliance exceeds 99% after fine-tuning.

## opp-capture Integration

`src/adapters/opp_capture.py` demonstrates how F4 would plug into the opp-capture Analyst pipeline. It wraps all F4 internals behind a single class:

```python
from src.adapters.opp_capture import F4Adapter

f4 = F4Adapter(model_id="arn:aws:bedrock:us-east-1:165286508758:imported-model/pxi20ybyyh5t")
result = f4.filter(combined_text)
# result: {"black": [...], "red": [...], "green": [...], "blue": [...], "unparsed_chunks": int, "total_chunks": int}
```

In opp-capture's `Analyst.analyze_and_evaluate()`, this runs after text extraction and **before** the Sonnet summarization call. Black flags short-circuit immediately (no LLM cost); red/green/blue flags merge into the final `Decision` object alongside the existing evaluator output. See the module docstring for the full integration example.

## Model

**Llama 3.2 3B-Instruct**, selected for:

- **Size:** 3B parameters — fast and cheap, fits a single Bedrock Custom Model Unit
- **Instruction-following:** IFEval 73.93%, best in class at 3B. Critical because Bedrock Custom Model Import has no constrained generation support.
- **Bedrock compatibility:** Proven import path, Llama architecture supported

Fine-tuned with LoRA (rank 16, alpha 32) on ~493 real RFP chunks labeled via Claude distillation (Sonnet labeling + Opus validation). Trained on SageMaker ml.g6.xlarge, 5 epochs, max sequence length 2048. Adapters merged back into base model and exported as HF safetensors for Bedrock import.

### Evaluation (Run 8 — Current)

Held-out test set, 61 chunks, 7 flags:

| Metric | Value |
|--------|-------|
| F1 | 88.7% |
| Precision | 92.2% |
| Recall | 85.5% |
| Format compliance | 98.4% |

Per-flag breakdown and full evaluation history (8 runs) in [evaluation_results.md](.development-notes/notes/evaluation_results.md).

## Flag Taxonomy

7 flags detected from individual RFP chunks:

| Tier | Flags | Meaning |
|------|-------|---------|
| Red | `lpta_source_selection`, `small_business_set_aside` | Negative indicators, configurable filter threshold |
| Green | `agile_methodology`, `oral_presentation` | Positive signals |
| Blue | `8a_set_aside`, `sdvosb_set_aside`, `hubzone_set_aside` | Informational (socioeconomic set-asides) |

No black (fast-fail) flags in the current model. Decision logic supports them — `lpta_source_selection` is a candidate for reclassification.

Full definitions and rationale for kept/dropped flags in [collated-flag-set.md](.development-notes/notes/collated-flag-set.md).

## Project Structure

```
src/
  chunking/chunker.py        # Word-based text chunking (~512 tokens)
  decision/engine.py         # Algorithmic filter decision (black/red threshold)
  domain/                    # Entities, parsing, protocols, taxonomy (7 flags)
  frontend/                  # Gradio demo app + PDF/DOCX extraction
  inference/bedrock.py       # BedrockFlagDetector + system prompt
  pipeline/filter.py         # F4Pipeline orchestrator
  rag/                       # ChromaDB store + retriever (built, not active)

training/
  train.py                   # LoRA fine-tuning (SFTTrainer)
  merge_and_export.py        # Merge LoRA -> base, export safetensors
  check_token_lengths.py     # Token length distribution check

evaluation/
  evaluate.py                # Per-flag precision, recall, format compliance

scripts/
  build_training_set.py      # Build train/eval/test from validated corpus
  test_bedrock_live.py       # Bedrock endpoint smoke test
  sonnet_label_real_data.py  # Sonnet labeling pipeline
  opus_validate_labels.py    # Opus validation of Sonnet labels
  opus_label_test_set.py     # Opus test set labeling
  opus_relabel_training_data.py  # Opus two-phase re-labeling

tests/                       # 164 tests, 89% coverage
data/                        # Active splits + validated corpus archive
```

## Development

```bash
uv sync
pytest --cov
ruff check . && ruff format .
```

Training (SageMaker ml.g6.xlarge):
```bash
uv sync --group training --group dev
uv run python training/train.py --max-seq-length 2048
uv run python training/merge_and_export.py
```

Evaluation (SageMaker):
```bash
PYTHONPATH=. uv run python evaluation/evaluate.py
```

Bedrock smoke test (local, Alt account creds):
```bash
uv run python scripts/test_bedrock_live.py \
  --model-arn "arn:aws:bedrock:us-east-1:165286508758:imported-model/pxi20ybyyh5t"
```

## Future Development

- **Chunk size tuning:** 512 tokens was conservative (assumed RAG context sharing). Larger chunks may improve recall.
- **Specialist models:** Group flags by detection strategy (set-asides, evaluation criteria, document-level context) instead of one generalist.
- **Pipeline-integrated data gathering:** Label chunks passively via the live opp-capture pipeline for higher-quality training data.
- **RAG revisited:** Disabled for the generalist model (too noisy across 30+ flag types). May be viable with specialist models and scoped stores.

## References

- [Architectural Decision Record](.development-notes/notes/final_adr.md)
- [Evaluation Results](.development-notes/notes/evaluation_results.md)
- [Bedrock Deployment Log](.development-notes/notes/bedrock-deployment.md)
- [Flag Set & Rationale](.development-notes/notes/collated-flag-set.md)
- [Retrospective](.development-notes/notes/retrospective.md)
