# F4 Plugin — Session Handoff

## What This Project Is

F4 (Flexion Fast Fail Filtering) is a Python library that screens government RFP text for disqualifying flags before expensive LLM analysis. It replaces string-based keyword matching in Flexion's opp-capture pipeline with a fine-tuned Llama 3.2 3B model deployed to AWS Bedrock Custom Model Import.

The library is standalone — opp-capture imports it via adapter behind an existing port. If F4 disappears, opp-capture keeps working.

## What's Been Built (Working)

**Full pipeline** (`src/pipeline/filter.py`): `f4.filter(text)` → chunk text → concurrent Bedrock inference → parse/retry → deduplicate flags → algorithmic decision → `FilterResult`.

**7 active flags** the model detects (reduced from an original set of 30+ after iterative evaluation):
- Red: `lpta_source_selection`, `small_business_set_aside`
- Blue: `8a_set_aside`, `sdvosb_set_aside`, `hubzone_set_aside`
- Green: `agile_methodology`, `oral_presentation`

No black (fast-fail) flags remain in the current model — all former black flags were dropped because they required wider document context or had insufficient training data. The user is considering reclassifying `lpta_source_selection` as black.

**Key components:**
- `src/domain/` — entities, parsing (with retry/leniency for none/comma-separated), taxonomy (7 flags), FlagDetector protocol
- `src/chunking/chunker.py` — word-based approximate chunking (~1.3 tokens/word). Replaced HF tokenizer to avoid gated model auth dependency. See bedrock-deployment.md §5.
- `src/inference/bedrock.py` — BedrockFlagDetector adapter, system prompt embedded
- `src/decision/engine.py` — algorithmic decision: black flag → FILTER, red flags ≥ threshold → FILTER. Threshold defaults to 999 (effectively disabled).
- `src/rag/` — ChromaDB RAG store + retriever. Code is intact but **currently not active** — the model was trained and evaluated without RAG. Pipeline accepts `rag_store=None` (default). Not ruled out for future use.
- `src/frontend/` — Gradio app with PDF/DOCX upload, calls real pipeline via Bedrock. `--share` tunnel + password auth. Recently tested and working.

**Training infrastructure** (runs on SageMaker ml.g6.xlarge):
- `training/train.py` — LoRA fine-tuning with SFTTrainer
- `training/merge_and_export.py` — merge LoRA → base, export safetensors. Includes config.json fix for Bedrock compatibility (rope_parameters→rope_scaling, transformers version downgrade). See bedrock-deployment.md §6 for the full RoPE debugging story.
- `training/check_token_lengths.py` — token length distribution check
- `evaluation/evaluate.py` — flag precision, chunk recall, format compliance, per-flag breakdown, no_flag hallucination stats

**Tests:** 164 passing, 89% coverage. All model/Bedrock calls mocked.

## Current Model Performance (Run 8 — Latest)

Held-out test set, 61 chunks, 7 flags:
- F1: 88.7%, Precision: 92.2%, Recall: 85.5%, Format compliance: 98.4%
- Strongest: oral_presentation (100/100), hubzone (100/100)
- Weakest: small_business_set_aside (77% precision, 3 FP)
- no_flag hallucination: 37.5% (3/8 no_flag chunks got false flags)

Full evaluation history (8 runs) is in `.development-notes/notes/evaluation_results.md`. The progression tells a story: synthetic data → real data → iterative data cleaning → flag reduction → current performance.

## Training Data Pipeline

Source: ~958 real RFPs labeled by Claude (Sonnet first pass → Opus validation). The validated corpus is `data/archive/opus_validated_real.jsonl` (4159 records, 3779 usable).

Current active splits (7-flag subset, built by `scripts/build_training_set.py`):
- `data/train.jsonl` (493 examples)
- `data/eval.jsonl` (61)
- `data/test.jsonl` (61)

`data/archive/test_reserve.jsonl` (501 chunks) — Opus-labeled with a stricter prompt. Potentially useful if filtered to the current 7 flags.

Labeling scripts in `scripts/` are prefixed by which model runs them:
- `sonnet_label_real_data.py` — initial Sonnet labeling
- `opus_validate_labels.py` — Opus reviews Sonnet labels
- `opus_label_test_set.py` — Opus labels test set directly
- `opus_relabel_training_data.py` — Opus two-phase re-labeling pipeline (chunk + self-validate)

## What Was Just Cleaned Up (This Session)

1. **Taxonomy aligned to 7 active flags** — `taxonomy.py` trimmed from 19→7, `collated-flag-set.md` Keep/Drop sections updated with reasons for each demotion, all tests updated.

2. **Scripts reorganized:**
   - Renamed labeling scripts with model prefix (sonnet_/opus_)
   - Archived one-time pipeline scripts to `archive/scripts/` (supplement_rare_flags, clean_zero_recall_flags, oversample_cleaned_flags, rewrap_rag_context, populate_rag)
   - Removed: bedrock_token_limit.py (documented in bedrock-deployment.md), check_token_lengths.py (duplicate of training/), split_test.py (superseded), generate_data.py (superseded by real data pipeline)

3. **Data cleaned:**
   - Archived `data/synthetic/` and `data/chromadb/` to `archive/data/`
   - Purged stale files from `data/archive/` (old splits, sonnet_labeled_real, rag_exemplars, test-p-override) — kept only opus_validated_real.jsonl and test_reserve.jsonl
   - Removed stale `training/evaluate.py` (duplicate of evaluation/evaluate.py, which has extra no_flag hallucination stats)

4. **Misc:** Moved `final_adr.md` into `.development-notes/notes/`, updated references in README and checklist. Removed empty `infra/terraform/` and redundant `data/.gitkeep`.

## Repo Structure (Post-Cleanup)

```
src/
  chunking/chunker.py        — word-based text chunking
  decision/engine.py         — algorithmic filter decision (black/red threshold)
  domain/                    — entities, parsing, protocols, taxonomy (7 flags)
  frontend/                  — Gradio app + text extraction
  inference/bedrock.py       — BedrockFlagDetector + system prompt
  pipeline/filter.py         — f4.filter() orchestrator
  rag/                       — ChromaDB store + retriever (code present, not active)

scripts/
  build_training_set.py      — build train/eval/test from opus_validated_real.jsonl
  debug_pipeline.py          — per-chunk debug output
  sonnet_label_real_data.py  — Sonnet labeling pipeline
  opus_label_test_set.py     — Opus test set labeling
  opus_validate_labels.py    — Opus validation of Sonnet labels
  opus_relabel_training_data.py — Opus two-phase re-labeling
  test_bedrock_live.py       — live Bedrock smoke test
  train.py                   — LoRA fine-tuning (SageMaker)
  system-prompt.md           — system prompt source of truth

training/                    — SageMaker training scripts (train, merge, token check)
evaluation/evaluate.py       — evaluation with per-flag + no_flag hallucination stats
tests/                       — 164 tests, 89% coverage

data/
  train.jsonl, eval.jsonl, test.jsonl  — active 7-flag splits
  archive/opus_validated_real.jsonl    — full validated corpus (source for rebuilding splits)
  archive/test_reserve.jsonl           — 501-chunk Opus-labeled test set (strict prompt)

.development-notes/
  project-checklist.md                 — master checklist with completion status
  notes/collated-flag-set.md           — full flag universe + Keep/Drop rationale
  notes/evaluation_results.md          — 8 runs of eval results with commentary
  notes/bedrock-deployment.md          — deployment log including RoPE debugging
  notes/final_adr.md                   — architectural decision record
  notes/presentation-notes.md          — presentation ideas and talking points
  notes/test-labeling-prompt.md        — evolved labeling prompt
  plans/                               — historical implementation plans

archive/                     — stale scripts and data (recoverable if needed)
```

## Next Steps

1. **Review the project checklist** (`.development-notes/project-checklist.md`) — assess what's done, what's incomplete, and what's out of scope for the presentation.
2. **Prepare a 10-minute presentation** of what's been implemented. Key references:
   - `.development-notes/notes/presentation-notes.md` — existing talking points and ideas
   - `.development-notes/notes/evaluation_results.md` — the iterative improvement story
   - `.development-notes/notes/bedrock-deployment.md` — the RoPE debugging story
   - The Gradio demo is live and working for a live walkthrough
3. The checklist has unchecked items (TextGrad, Terraform, observability, delta eval, Tom Willis data) — decide what to present as "implemented" vs. "designed but deferred" vs. "future work."

## Important Context

- **Academic project** — this is a learning exercise. Understanding why/how matters more than speed.
- **Git: user handles all git operations.** Never commit, push, or make git write operations.
- **Bedrock model** is deployed on the Alt/cohort account. Model ARN in checklist. Cold start ~1-2 min after idle.
- **Tom Willis** maintains opp-capture and has real RFP decision data — not yet integrated.
- **opp-capture repo** is at `/Users/travisblount-elliott/Repos/flexion-opp-capture` — F4 plugs in behind `OpportunityEvaluatorGateway` port.
