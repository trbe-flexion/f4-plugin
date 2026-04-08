# F4: Flexion Fast Fail Filtering

A standalone service that detects categories of flags in RFP text using a fine-tuned small model. Designed as a plug-in for the opp-capture pipeline.

The service accepts extracted RFP text via API, chunks it, runs each chunk through a LoRA fine-tuned model augmented with RAG context, parses detected flags, deduplicates across chunks, and returns a JSON response with flags and a PASS/FAIL/REVIEW decision.

See the [Architectural Decision Record](final_adr.md) for full design details.

## Project Structure

```
f4-plugin/
├── src/
│   ├── api/            # Service API layer
│   ├── chunking/       # Text chunking logic
│   ├── inference/      # Bedrock model calls
│   ├── rag/            # ChromaDB retrieval
│   ├── decision/       # Flag aggregation and PASS/FAIL/REVIEW logic
│   └── frontend/       # Gradio demo app
├── tests/              # pytest test suite
├── data/               # Training and evaluation datasets
├── training/           # Fine-tuning scripts
├── evaluation/         # Eval scripts and results
└── infra/
    ├── terraform/      # Bedrock, ECR, IAM
    └── docker/         # Container definitions
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
