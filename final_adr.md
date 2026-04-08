# F4: Flexion Fast Fail Filtering

## Status

Proposed/Planned

## Context

Flexion's opp-capture pipeline screens government RFPs for business fit. The current Screener stage uses string-based keyword matching — effective for obvious cases but blind to nuance. "Standard software lifecycle development" means waterfall, but there's no keyword for that. The Analyzer stage uses a large cloud LLM (Claude via Bedrock) for deeper evaluation, but that's expensive and slow per-call.

I want to build a Python library that replaces the string-based filter with a fine-tuned small model that detects categories of flags in RFP text. The library (F4 — Flexion Fast Fail Filtering) will be a plug-in that opp-capture can adopt or ignore. If my library disappears tomorrow, opp-capture keeps working. This is a final project for the LLMs in Production course, demo April 20.

Tom Willis maintains opp-capture and has historical decision data and real RFP examples available.

## Plan Outline

1. Create repo (`f4-plugin`), project structure, infra scaffolding
2. Finalize this ADR
3. Research model selection (Bedrock Custom Model Import compatibility is the gate)
4. Finalize flag set and output format
5. Generate synthetic training/validation data via Claude (distillation)
6. Source and manually label real RFP chunks for test set
7. LoRA fine-tuning
8. Set up RAG — ChromaDB with flag definitions, examples, and real RFP language from Tom
9. Deploy fine-tuned model to Bedrock Custom Model Import
10. Build library layer (public `filter()` interface, chunking, flag aggregation, decision logic)
11. Prompt optimization via TextGrad on SageMaker
12. Gradio frontend for manual RFP upload and demo
13. Evaluation (precision/recall on synthetic and real data, opp-capture adapter for delta comparison against current filtering)
14. Documentation and presentation prep

## Key Decisions

### Deployment Target

Bedrock Custom Model Import. The model deploys as a Bedrock endpoint — no GPU instances to manage and native to the AWS ecosystem opp-capture already runs in. Pricing is per-minute based on Custom Model Units (CMUs), billed in 5-minute windows at $0.05718/CMU/min (US regions). A Llama 3.1 8B requires 2 CMUs; a smaller 3B model likely requires 1 (determined at import time). Bedrock scales to zero automatically after 5 minutes of inactivity. Cold start on scale-up is "tens of seconds" depending on model size — acceptable for a batch pipeline. Opp-capture already speaks to Bedrock via LiteLLM, so my model is just another model ID in the config.

Consequences: Giving up constrained generation (Outlines, GBNF grammars). Bedrock does not support inference-time sampling constraints on custom imported models, and structured output / JSON mode is limited to foundation models. I am betting that model selection, fine-tuning on the correct output format, and prompt optimization can produce predictably formatted output without enforcement. The output format is simple (see Output Format below), which reduces this risk. A single retry on parse failure provides a safety net.

### Output Format

The model outputs detected flags in a simple parseable format (CSV or similar), not JSON. I am hoping a line like `waterfall_methodology,fast-fail` is easy to reliably prompt, and it would be trivially parsed by the application code. The service layer — my application code, not the model — constructs the final JSON response containing both flags and the algorithmic decision. This means structured JSON output is guaranteed regardless of model behavior.

### Architectural Boundary

F4 is a Python library in its own repo (`f4-plugin`) with its own infrastructure code (Terraform for Bedrock Custom Model Import, IAM roles). Opp-capture imports F4 and calls `f4.filter(text)`, which returns a dict with detected flags, a PASS/FAIL/REVIEW decision, and processing metadata. The pipeline logic (chunking, RAG, inference, parsing, deduplication, decision) runs in-process — F4 is not a deployed web service.

On the opp-capture side, the integration surface is minimal: a new adapter behind an existing port, plus a config toggle. Opp-capture may want to disable F4 at times to avoid costs when the pipeline doesn't need it. If F4 gets deleted, opp-capture loses nothing. This follows opp-capture's own OEA principles — no invasive species.

Consequences: I own my own infra (Bedrock model import, IAM) but not running compute. Clean separation. The opp-capture adapter is minimal — a new adapter behind an existing port — and is part of the evaluation plan for delta comparison.

### Library Contract

Thick contract. `f4.filter(text)` accepts RFP text and returns a dict containing detected flags, a PASS/FAIL/REVIEW decision, and processing metadata. The flow inside the library: chunk the text → model detects flags per chunk → parse and retry once on failure → deduplicate flags across chunks → apply algorithmic decision logic → return result.

The response includes an `unparsed_chunks` count so the caller knows if any chunks failed parsing after retry.

### Chunking Strategy

RFPs will be chunked at roughly 512-1024 tokens as a starting point, tunable based on testing. Chunks will overlap to avoid missing context at boundaries. Each chunk is independently sent to the model for flag detection. Flags are deduplicated across all chunks — either a flag is present in the RFP or it isn't.

Per-chunk retry: if the model output fails to parse, retry that chunk once. If it still fails, discard the chunk and increment the `unparsed_chunks` counter. The full RFP is not retried.

### Flag Taxonomy

Collated from my previous biz-dev work and opp-capture's existing evaluation config. Four severity tiers: fast-fail, red, green, and blue (informational). Full flag set documented in `collated-flag-set.md`. The flag set is finalized in draft and subject to manual review.

### Decision Logic

TBD. The algorithmic rules mapping flag combinations to PASS/FAIL/REVIEW outcomes are configurable and can be adjusted in code without retraining. This is a business policy conversation that may happen after the final presentation, in collaboration with biz-dev. The configurability is a feature — different organizations could define their own flag sets and decision rules.

### Model Selection

TBD pending research. The gating constraint is Bedrock Custom Model Import, which supports a limited range of models.

Candidates under consideration:

- Llama 3.2 1B/3B — smallest Llama, fast inference, lowest Bedrock cost
- Qwen2.5-3B — prior experience from HW9/HW10
- Mistral 7B — strong instruction-following
- Llama 3.1 8B / Qwen2.5-7B — fallback if 3B models underperform

For a narrow classification task (detect known flags from ~512-1024 token chunks), a 3B model is likely sufficient, and the smaller/cheaper the model, the more likely it will compare favorably against a more expensive OOTB LLM like Claude.

After LoRA fine-tuning, adapters must be merged back into the base model and exported as Hugging Face safetensors format for Bedrock Custom Model Import.

### Datasets

Data Source:
- Synthetic chunks: generated by Claude (Opus 4.6) via distillation — the small model learns to replicate the large model's judgment on flag detection. RFP-style text chunks labeled with correct flags, including examples with no flags, single flags, multiple flags, adversarial examples (text that resembles a flag but isn't), and ambiguous text. Volume TBD.
- Real RFP chunks: extracted from actual RFPs, manually labeled with expected flags. Tests generalization beyond synthetic data — different writing styles, different flag distributions. This is the test set.

Train/Eval/Test Split:
- Train Set Data: 80% of Synthetic Set.
- Eval Set: 20% of Synthetic Set
- Test Set: Small set of real RFP chunks.

### Evaluation Strategy

Finetuning evaluation across training epochs and on the held-out eval set will depend on two metrics.
- Flag Level Precision: Of the flags the model detected, how many were actually present? Low precision on a fast-fail flag kills a good opportunity (missed potential revenue).
- Chunk Level Recall: Of the flags actually present, how many did the model detect? Low recall lets a bad RFP through (wasted effort).

An additional evaluation will occur after plugging into the opp-capture pipeline.
- Delta against current filtering: run both F4 and opp-capture's existing string/metadata-based filtering on the same set of RFPs during a production run, and compare. Did F4 catch flags that string matching missed? Did the current system get false-positives that F4 parsed correctly?

### RAG Purpose

ChromaDB vector store containing flag definitions, example passages, and real RFP language sourced from Tom. The RAG serves four purposes:

1. Patch gaps: if training missed certain flag expressions, add example passages post-training without retraining
2. Extensibility: add entirely new flag types via definitions and examples in the vector store. The model receives these as in-context examples at inference time and can detect the new flags without retraining. Retraining can happen later if enough new flags accumulate. Note: this capability depends on model selection — a 7B model is much more likely to generalize to unseen flag IDs via in-context learning than a 3B model. Validate early with a held-out flag test.
3. Evaluation: measure whether RAG-augmented inference outperforms base fine-tuned inference (stretch goal)
4. Real-world grounding: real RFP language in the RAG gives the model context that synthetic training data may miss

At inference time, each chunk is embedded and the top-k most similar flag definitions and example passages are retrieved from ChromaDB and prepended to the prompt as context. The value of k is tunable — too few and relevant flags are missed, too many and the prompt is diluted with noise and inflated context size (on top of the chunk's own 512-1024 tokens).

### Prompt Optimization

TextGrad, applied after fine-tuning but before Bedrock deployment. TextGrad treats the system prompt as a trainable variable and uses an LLM backward engine to generate critiques and rewrites. The forward engine is the fine-tuned model loaded locally on SageMaker; the backward engine is Claude Opus via Bedrock (SageMaker can call Bedrock with the right IAM role). Running the forward engine locally avoids CMU billing and cold starts during the many forward passes TextGrad requires per optimization step. The optimized prompt is then used when the model is deployed to Bedrock.

If it doesn't help, that's a valid finding worth documenting.

### Frontend

Gradio app with a `gr.File()` upload component. Accepts PDF/DOCX, extracts text (duplicating opp-capture's extractors for demo convenience), calls `f4.filter()`, and displays the flag report and decision. This is a demo surface — the real integration point is the library.

Consequences: text extraction is duplicated from opp-capture. The library itself accepts pre-extracted text. For production use via opp-capture, text extraction happens on their side.

### Text Input

`f4.filter()` accepts pre-extracted text, not files. Opp-capture already handles text extraction (pdfplumber, docx). The Gradio frontend handles extraction locally for demo purposes only.

### Observability

The library will log per-batch running time, token counts (from Bedrock response metadata where available, estimated client-side via the model's tokenizer otherwise), and inference cost. This is essential — without it, there's no way to empirically compare F4 against the Claude baseline (see below) and justify or invalidate the fine-tuned model approach.

### Cost vs. Claude Baseline

The obvious alternative is skipping the fine-tuned model entirely and sending each RFP to Claude (already available via Bedrock in opp-capture) with a flag-detection prompt. This would be simpler — no training data, no fine-tuning, no chunking, no RAG, and Claude's large context window handles full documents without chunking. The cost case for F4 over Claude depends on batch volume, document size, and how many sources opp-capture queries. At low volume, Claude is likely cheaper and simpler. At high volume, F4's time-based pricing amortizes better than Claude's per-token pricing, especially for large documents.

This is one reason F4 tracks running time and token usage per batch (see Observability above) — so the comparison can be made with real numbers, not estimates.

## Consequences

This project touches nearly every skill from the course: LoRA fine-tuning, RAG, prompt optimization, constrained output via prompting, evaluation, Bedrock deployment, and frontend. The risk is breadth — there's a lot to build in two weeks.

The configurable flag set and decision logic make this potentially productizable beyond Flexion. There's no reason another organization couldn't define their own flags and use this library for their own RFP screening.

The standalone architecture means I can develop and demo independently of opp-capture's release cycle. The opp-capture adapter is needed for the delta evaluation but is minimal work given opp-capture's existing port/adapter pattern.

## Rubric Alignment

Model & Inference: LoRA fine-tuned model, synthetic training data, real RFP test set, RAG for extensibility, TextGrad prompt optimization.

Production Environment: Bedrock Custom Model Import. Library with own infra (Terraform for model import, IAM). Opp-capture adapter for delta evaluation. Observability for cost comparison against Claude baseline.

Inference Pipeline: `f4.filter(text)` → chunking → RAG retrieval → model inference → parse/retry → flag deduplication → algorithmic decision → result dict.

Documentation: This ADR, plus technical docs on model selection, training data generation, evaluation results, and deployment.

Demo: Gradio frontend — upload an RFP, get a flag report and decision. Delta comparison against current string/metadata filtering.

## Notes

F4 = Flexion Fast Fail Filtering.

Document edited with LLM aid.
