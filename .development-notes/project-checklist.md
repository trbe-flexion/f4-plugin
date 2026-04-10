# F4 Project Checklist

Checklist extracted from ADR (/Users/travisblount-elliott/Repos/f4-plugin/final_adr.md)

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

## 6. Real RFP Test Set *(deferred — using a held-out subset of synthetic data as test set for now)*
- [x] Hold-out subset of synthetic data for test set.
- [ ] Source real RFP chunks from Tom Willis
- [ ] Manually label with expected flags

## 7. RAG Setup *(deferred — RAG context baked into training data; live ChromaDB for inference pipeline later)*
- [ ] Set up ChromaDB vector store
- [ ] Populate with flag definitions and examples (`data/rag_seeds.jsonl`)
- [ ] Add real RFP language from Tom
- [ ] Tune top-k retrieval parameter

## 8. LoRA Fine-Tuning *(training scripts: `scripts/train.py`, `scripts/merge_and_export.py`)*
- [x] Set up training environment (SageMaker ml.g6.xlarge)
- [ ] Fine-tune on synthetic training set *(in progress — training running)*
- [ ] Evaluate on synthetic eval set (flag precision, chunk recall)
- [ ] Merge LoRA adapters back into base model
- [ ] Export as HF safetensors

## 9. Bedrock Deployment
- [ ] Terraform for Bedrock Custom Model Import + IAM
- [ ] Import merged model to Bedrock
- [ ] Verify endpoint inference

## 10. Library Layer
- [x] Implement `f4.filter(text)` public interface (`src/pipeline/filter.py`)
- [x] Text chunking (token-boundary, configurable overlap) (`src/chunking/chunker.py`)
- [ ] RAG retrieval per chunk *(deferred — depends on ChromaDB setup in step 7)*
- [x] Concurrent Bedrock inference (bounded ThreadPoolExecutor) (`src/pipeline/filter.py`)
- [x] Parse/retry logic (1 retry, unparsed tracking) (`src/domain/parsing.py`, `src/pipeline/filter.py`)
- [x] Flag deduplication across chunks (`src/pipeline/filter.py`)
- [x] Algorithmic decision logic (black flag filter + configurable red threshold) (`src/decision/engine.py`)
- [x] Domain: FilterResult entity, FlagDetector protocol, flag taxonomy (`src/domain/`)
- [ ] Observability: timing, token counts, cost logging
- [ ] BedrockFlagDetector adapter *(depends on Bedrock deployment in step 9)*

## 11. Prompt Optimization (TextGrad)
- [ ] Set up TextGrad on SageMaker
- [ ] Forward engine: fine-tuned model (local)
- [ ] Backward engine: Claude Opus via Bedrock
- [ ] Optimize system prompt
- [ ] Document results (even if it doesn't help)

## 12. Gradio Frontend
- [ ] File upload component (PDF/DOCX)
- [ ] Text extraction for demo
- [ ] Call `f4.filter()` and display flag report + decision
- [ ] `share=True` tunnel + password auth for demo

## 13. Evaluation
- [ ] Precision/recall on synthetic eval set
- [ ] Precision/recall on real RFP test set
- [ ] Delta comparison: F4 vs current string-based filtering on same RFPs
- [ ] Opp-capture adapter for delta comparison
- [ ] Cost comparison vs Claude baseline (using observability data)

## 14. Documentation & Presentation
- [ ] Technical docs (model selection, training data, eval results, deployment)
- [ ] Presentation prep
