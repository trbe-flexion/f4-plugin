<img src="assets/f4_logo-1.png" alt="F4 Logo" width="48" align="left" style="margin-right: 12px;">

# F4: Flexion Fast Fail Filtering

A Python library that detects categories of flags in RFP text using a fine-tuned small model. Designed as a plug-in for the opp-capture pipeline.

`f4.filter(text)` accepts extracted RFP text, chunks it, runs each chunk through a LoRA fine-tuned model augmented with RAG context, parses detected flags, deduplicates across chunks, and returns a dict with flags, a FILTER/REVIEW decision, and processing metadata.

See the [Architectural Decision Record](final_adr.md) for full design details.

## Project Structure

```
f4-plugin/
├── src/
│   ├── pipeline/       # Public filter() interface and orchestration
│   ├── chunking/       # Text chunking logic
│   ├── inference/      # Bedrock model calls
│   ├── rag/            # ChromaDB retrieval
│   ├── decision/       # Flag aggregation and FILTER/REVIEW logic
│   └── frontend/       # Gradio demo app
├── tests/              # pytest test suite
├── data/               # Training and evaluation datasets
├── training/           # Fine-tuning scripts
├── evaluation/         # Eval scripts and results
└── infra/
    └── terraform/      # Bedrock Custom Model Import, IAM
```

## Setup

```bash
uv sync
pre-commit install
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
