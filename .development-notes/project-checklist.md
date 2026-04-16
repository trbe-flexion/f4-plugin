# F4 Project Checklist

Checklist extracted from ADR (.development-notes/notes/final_adr.md)

## 1. Repo & Structure
- [x] Create repo (`f4-plugin`)
- [x] Project structure and infra scaffolding

## 2. ADR
- [x] Finalize ADR

## 3. Model Selection
- [x] Research Bedrock Custom Model Import compatibility
- [x] Select base model

## 4. Flag Set & Output Format
- [x] Review and finalize flag set (`collated-flag-set.md`)
- [x] Confirm output format (CSV-like lines)

## 5. Synthetic Training Data
- [x] Design data generation prompts for Claude distillation (`scripts/generate_data.py`)
- [x] Generate synthetic RFP chunks with flag labels (`data/train.jsonl`, `data/eval.jsonl`)
- [x] Include: no flags, single flags, multiple flags, adversarial, ambiguous examples (`data/train.jsonl`)
- [x] 80/20 train/eval split (`data/train.jsonl`, `data/eval.jsonl`)

## 6. Real RFP Training & Test Data
- [x] Hold-out subset of synthetic data for test set
- [x] Label real RFP data with Claude distillation (`data/sonnet_labeled_real.jsonl`, ~4040 chunks from 958 RFPs)
  - Extract text from real RFP files
  - Chunk and send to Claude (Sonnet via Bedrock) for flag labeling
  - Output as labeled JSONL
- [x] Validation pass: Opus reviews all Sonnet-labeled chunks (`data/opus_validated_real.jsonl`)
  - 3779 usable, 261 dropped (mislabeled, ambiguous, or substantive flag concerns)
  - ~13% error rate confirmed — Opus fixed or dropped problematic examples
- [x] Supplement rare flags with Opus-generated synthetic examples (`scripts/supplement_rare_flags.py`)
  - hubzone (15→30), wosb (17→34), budget_too_low (23→46), 8a (24→48)
  - Seeded with random real examples per flag for tone/style matching
- [x] Decide: detect `onsite_madison` via string match post-filter on `onsite_required` (not model)
- [x] Build train/eval/test splits + RAG seeds (`scripts/build_training_set.py`)
  - 2 RAG seeds per flag reserved (30 total, excluded from splits)
  - off_the_shelf_software capped at 100; negatives capped 1:1 with positives
  - Stratified split by rfp_id (80/10/10): train 1536 / eval 193 / test 191
- [x] Retrain on real data (`data/train.jsonl`, `data/eval.jsonl`)
- [ ] ~~Source manually labeled test set from Tom Willis~~ *(deferred — not available)*

## 7. RAG Setup *(ChromaDB store + retriever built; currently disabled — added noise to prompts without improving performance)*
- [x] Set up ChromaDB vector store (`src/rag/store.py`)
- [x] Populate script for flag definitions and examples (`scripts/populate_rag.py`)
- [x] Retriever with training-data-matching context format (`src/rag/retriever.py`)
- [x] Wired into pipeline as optional dependency (`src/pipeline/filter.py`)
- [ ] ~~Add real RFP language from Tom~~ *(deferred — RAG disabled)*
- [ ] ~~Tune top-k retrieval parameter~~ *(deferred — RAG disabled)*

## 8. LoRA Fine-Tuning *(training scripts: `training/train.py`, `training/merge_and_export.py`)*
- [x] Set up training environment (SageMaker ml.g6.xlarge)
- [x] Fine-tune on synthetic training set (3 epochs, eval loss 0.53, token accuracy 86.5%)
- [x] Evaluation script (`training/evaluate.py`) — flag precision, chunk recall, format compliance, per-flag breakdown
- [x] Run evaluation on test set (F1: 94.1%, precision: 97.3%, recall: 91.1%, compliance: 100%). Results in `.development-notes/notes/evaluation_results.md`.
- [x] Merge LoRA adapters back into base model
- [x] Export as HF safetensors (uploaded to s3://trbe-f4-finetuned-model/)

**Decision:** Proceeding with current fine-tuned model as-is. Results are strong enough to build the full library against. Further tuning (TextGrad prompt optimization, additional training data, onsite_required/brownfield recall gaps) is plug-and-play — swap in improved models later without changing library code.

### 8b. Retrain with Realistic RAG *(plan: `.development-notes/plans/retrain-plan.md`)*
- [x] Rewrap training data with real ChromaDB retrieval (`scripts/rewrap_rag_context.py`)
- [x] Retrain on SageMaker (v2: eval loss 0.514, token accuracy 87.0%)
- [x] Re-evaluate (v2: F1 83.0%, precision 91.6%, recall 75.9%, compliance 97.5%)
- [x] Re-merge, re-upload to S3, re-import to Bedrock
- [x] Fix config.json: `rope_parameters` → `rope_scaling` for Bedrock compatibility (see `bedrock-deployment.md` section 6)
- [x] Re-test with real RFP — format compliance 100%, but low recall on real RFP text (domain gap)
- [x] Parser leniency: handle `none`/`None`, comma-separated flags (`src/domain/parsing.py`)

## 9. Bedrock Deployment
- [ ] ~~Terraform for Bedrock Custom Model Import + IAM~~ *(deferred — deployed manually)*
- [x] Import merged model to Bedrock (Alt/cohort account)
- [x] Verify endpoint inference (4/5 smoke test, 100% format compliance)
- [ ] ~~Migrate to Main account~~ *(deferred — needs Main credentials from team)*

## How to Run

Tests (local, no GPU):
  uv sync
  pytest --cov

Training (SageMaker ml.g6.xlarge):
  uv sync --group training --group dev
  uv run python training/check_token_lengths.py
  uv run python training/train.py --max-seq-length 2048
  uv run python training/merge_and_export.py

Populate RAG store:
  uv run python scripts/populate_rag.py

Evaluation (SageMaker, after training — use PYTHONPATH=. for module resolution):
  PYTHONPATH=. uv run python training/evaluate.py                  # fine-tuned only (default)
  PYTHONPATH=. uv run python training/evaluate.py --base-only      # base model only
  PYTHONPATH=. uv run python training/evaluate.py --compare        # both + comparison table
  Results saved to evaluation/baseline.json and evaluation/finetuned.json

Gradio demo (local, needs AWS creds for Alt account):
  uv run python -m src.frontend --model-arn "arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:imported-model/3ffr95d8c4cc" --share --auth demo:demo
  Options: --no-rag (skip RAG), --max-workers N (concurrent Bedrock calls), --region REGION
  Note: first invocation after idle hits Bedrock cold start (~1-2 min)

Bedrock smoke test (local, Alt account creds):
  uv run python scripts/test_bedrock_live.py --model-arn "arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:imported-model/3ffr95d8c4cc"

Opp-capture integration:
  pip install git+ssh://git@github.com/trbe-flexion/f4-plugin.git
  RAG store auto-populates from rag_exemplars.jsonl on first init.

---

## 10. Library Layer
- [x] Implement `f4.filter(text)` public interface (`src/pipeline/filter.py`)
- [x] Text chunking (token-boundary, configurable overlap) (`src/chunking/chunker.py`)
- [x] RAG retrieval per chunk (wired into pipeline, optional via rag_store param)
- [x] Concurrent Bedrock inference (bounded ThreadPoolExecutor) (`src/pipeline/filter.py`)
- [x] Parse/retry logic (1 retry, unparsed tracking) (`src/domain/parsing.py`, `src/pipeline/filter.py`)
- [x] Flag deduplication across chunks (`src/pipeline/filter.py`)
- [x] Algorithmic decision logic (black flag filter + configurable red threshold) (`src/decision/engine.py`)
- [x] Domain: FilterResult entity, FlagDetector protocol, flag taxonomy (`src/domain/`)
- [ ] ~~Observability: timing, token counts, cost logging~~ *(deferred)*
- [x] BedrockFlagDetector adapter (`src/inference/bedrock.py`)

## 11. Prompt Optimization *(manual optimization done; TextGrad deferred)*
- [x] Manual prompt optimization through iterative evaluation (8 runs)
- [ ] ~~TextGrad automated optimization~~ *(deferred — learned that optimization likely needs to be flag-by-flag, not whole-prompt)*

## 12. Gradio Frontend
- [x] File upload component (PDF/DOCX, multiple files) (`src/frontend/app.py`)
- [x] Text extraction for demo (pdfplumber + python-docx) (`src/frontend/extraction.py`)
- [x] Multi-file concatenation matching opp-capture format
- [x] Call `f4.filter()` and display decision + pipeline logs
- [x] `share=True` tunnel + password auth support
- [x] Wire with real BedrockFlagDetector + RAG store for live demo (`src/frontend/__main__.py`)

## 13. Evaluation *(deferred — waiting for production deployment)*
- [x] Precision/recall on synthetic eval set (Run 1–4)
- [x] Precision/recall on real-data-derived test set (Run 5–8, best: F1 88.7%)
- [ ] ~~Delta comparison: F4 vs current string-based filtering on same RFPs~~ *(deferred)*
- [ ] ~~Opp-capture adapter for delta comparison~~ *(deferred)*
- [ ] ~~Cost comparison vs Claude baseline~~ *(deferred — needs observability data)*

## 14. Documentation & Presentation
- [ ] Technical docs (model selection, training data, eval results, deployment)
- [ ] Presentation prep

---

## TEMP: Copy-paste commands (delete when done)

```
PYTHONPATH=. uv run python scripts/label_test_set.py "/Users/travisblount-elliott/For Travis/Pursue" "/Users/travisblount-elliott/For Travis/Do Not Pursue" --exclude "Sources Sought" "Capability_Statement"
```
