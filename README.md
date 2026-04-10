<h1 align="center">
  <img src="assets/f4_logo-1.png" alt="F4 Logo" width="64">&nbsp;
  F4: Flexion Fast Fail Filtering
  &nbsp;<img src="assets/f4_logo-1.png" alt="F4 Logo" width="64">
</h1>

A Python library that screens government RFP text for disqualifying flags before expensive LLM analysis. F4 replaces string-based keyword matching with a fine-tuned small model (Llama 3.2 3B) augmented by RAG retrieval, delivering chunk-level flag detection at a fraction of the cost of a large model.

Designed as a standalone plug-in for the [opp-capture](https://github.com/trbe-flexion/flexion-opp-capture) pipeline. If F4 disappears, opp-capture keeps working.

See the [Architectural Decision Record](final_adr.md) for full design details.

## How It Works

```
f4.filter(text)
  → chunk text at token boundaries (configurable overlap)
  → retrieve relevant flag definitions from ChromaDB (RAG)
  → send chunks concurrently to fine-tuned model via Bedrock
  → parse model output (one flag name per line)
  → retry once on parse failure, discard on second failure
  → deduplicate flags across all chunks
  → apply decision logic (any black flag → filter)
  → return FilterResult(filter=True/False)
```

## Flag Taxonomy

19 flags across four tiers, detected from individual RFP chunks:

| Tier | Flags | Effect |
|------|-------|--------|
| Black | waterfall_methodology, off_the_shelf_software, no_custom_development, onsite_required, budget_too_low | Any one → filter out |
| Red | small_business_set_aside, brownfield, lpta_source_selection, marginal_short_duration | Configurable threshold |
| Green | agile_methodology, oral_presentation, design_exercise | Positive signal |
| Blue | large_team, 8a/wosb/edwosb/sdvosb/hubzone_set_aside, onsite_madison | Informational |

Full definitions in [collated-flag-set.md](.development-notes/collated-flag-set.md).

## Project Structure

```
f4-plugin/
├── src/
│   ├── domain/        # Entities, protocols, taxonomy, parsing
│   ├── pipeline/      # F4Pipeline orchestrator
│   ├── chunking/      # Token-boundary text chunking
│   ├── rag/           # ChromaDB store + retriever
│   ├── decision/      # FilterDecisionEngine
│   ├── inference/     # Bedrock adapter (planned)
│   └── frontend/      # Gradio demo app
├── scripts/
│   ├── train.py               # LoRA fine-tuning (SFTTrainer)
│   ├── merge_and_export.py    # Merge LoRA → safetensors for Bedrock
│   ├── check_token_lengths.py # Training data analysis
│   └── populate_rag.py        # Seed ChromaDB from rag_seeds.jsonl
├── tests/             # 111 tests, 96% coverage
├── data/              # Training, eval, test JSONL + RAG seeds
└── infra/             # Terraform for Bedrock Custom Model Import
```

## Setup

```bash
uv sync
pre-commit install
```

For training (SageMaker only):
```bash
uv sync --group training --group dev
```

## Usage

```python
from src.pipeline.filter import F4Pipeline
from src.rag.store import FlagRAGStore

# Auto-populates RAG store from rag_seeds.jsonl on first run
store = FlagRAGStore.get_or_init()

pipeline = F4Pipeline(
    flag_detector=my_bedrock_detector,  # implements FlagDetector protocol
    tokenizer=my_tokenizer,
    rag_store=store,
)

result = pipeline.filter(rfp_text)
print(result.filter)  # True = filter out, False = pass through
```

## Testing

```bash
pytest --cov
```

## Linting

```bash
ruff check .
ruff format .
```

## Training

On SageMaker (ml.g6.xlarge):
```bash
uv run python scripts/check_token_lengths.py
uv run python scripts/train.py --max-seq-length 2048
uv run python scripts/merge_and_export.py
```

## Evaluation

On SageMaker (after training):
```bash
uv run python evaluation/evaluate.py                  # fine-tuned model (default)
uv run python evaluation/evaluate.py --base-only      # base model baseline
uv run python evaluation/evaluate.py --compare        # both + side-by-side comparison
```

Outputs flag-level precision, chunk-level recall, format compliance, per-flag breakdown, and saves full results to `evaluation/`.

## Gradio Demo

```bash
uv run python -m src.frontend.app --share --auth user:password
```

Uploads PDF/DOCX files, extracts text, runs the full pipeline, and displays the decision with detailed logs.
