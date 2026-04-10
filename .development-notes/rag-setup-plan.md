# ChromaDB RAG Setup Plan

## Context

Checklist step 7. Training data has RAG context baked in (synthetic examples). The inference pipeline needs live retrieval so the model gets similar context at inference time as it saw during training. ChromaDB stores flag definitions and example RFP passages from data/rag_seeds.jsonl (190 records, 10 per flag, 19 flags). Real RFP language from Tom Willis can be added later.

Reference: HW8 ChromaDB implementation at archive/hw8/src/vectordb.py.

## Decisions

- Embedding model: intfloat/e5-base-v2 (same as HW8, no API keys, runs locally)
- Storage: PersistentClient for production, ephemeral Client for tests
- Distance metric: cosine
- Collection: get_or_create_collection (idempotent)
- Metadata per document: flag name
- Top-k: configurable, default 3

## Files Created

- src/rag/store.py — FlagRAGStore class (ChromaDB wrapper)
- src/rag/retriever.py — format_context for retrieved passages
- scripts/populate_rag.py — load rag_seeds.jsonl into ChromaDB
- tests/test_rag_store.py, tests/test_retriever.py

## Files Modified

- pyproject.toml — chromadb + sentence-transformers deps
- src/pipeline/filter.py — optional RAG retrieval before inference

## Steps

1. Save plan to .development-notes/
2. Add chromadb + sentence-transformers to pyproject.toml
3. FlagRAGStore: add_passages, query(top-k), count. PersistentClient or ephemeral.
4. Retriever: format_context matching "[Retrieved context]\n\nExample N:\n{passage}" format from training data
5. populate_rag.py: load rag_seeds.jsonl, idempotent
6. Wire into pipeline: optional rag_store on F4Pipeline, prepend context before inference
7. Tests with ephemeral ChromaDB, 80%+ coverage
8. Lint, verify, update checklist
