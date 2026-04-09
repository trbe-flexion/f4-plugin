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
- [ ] Review and finalize flag set (`collated-flag-set.md`)
- [ ] Confirm output format (CSV-like lines)

## 5. Synthetic Training Data
- [ ] Design data generation prompts for Claude distillation
- [ ] Generate synthetic RFP chunks with flag labels
- [ ] Include: no flags, single flags, multiple flags, adversarial, ambiguous examples
- [ ] 80/20 train/eval split

## 6. Real RFP Test Set
- [ ] Source real RFP chunks from Tom Willis
- [ ] Manually label with expected flags

## 7. LoRA Fine-Tuning
- [ ] Set up training environment (SageMaker)
- [ ] Fine-tune on synthetic training set
- [ ] Evaluate on synthetic eval set (flag precision, chunk recall)
- [ ] Merge LoRA adapters back into base model
- [ ] Export as HF safetensors

## 8. RAG Setup
- [ ] Set up ChromaDB vector store
- [ ] Populate with flag definitions and examples
- [ ] Add real RFP language from Tom
- [ ] Tune top-k retrieval parameter

## 9. Bedrock Deployment
- [ ] Terraform for Bedrock Custom Model Import + IAM
- [ ] Import merged model to Bedrock
- [ ] Verify endpoint inference

## 10. Library Layer
- [ ] Implement `f4.filter(text)` public interface
- [ ] Text chunking (512-1024 tokens, overlap)
- [ ] RAG retrieval per chunk
- [ ] Concurrent Bedrock inference (bounded concurrency)
- [ ] Parse/retry logic (1 retry, `unparsed_chunks` counter)
- [ ] Flag deduplication across chunks
- [ ] Algorithmic decision logic (configurable rules)
- [ ] Observability: timing, token counts, cost logging

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
