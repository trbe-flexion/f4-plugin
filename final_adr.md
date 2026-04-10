# F4: Flexion Fast Fail Filtering

## Status

Proposed/Planned

## Context

Flexion's opp-capture pipeline screens government RFPs for business fit. The current Screener stage uses string-based keyword matching — effective for obvious cases but blind to nuance. "Standard software lifecycle development" means waterfall, but there's no keyword for that. The Analyzer stage uses a large cloud LLM (Claude via Bedrock) for deeper evaluation, but that's expensive and slow per-call.

I want to build a Python library that replaces the string-based filter with a fine-tuned small model that detects categories of flags in RFP text. The library (F4 — Flexion Fast Fail Filtering) will be a plug-in that opp-capture can adopt or ignore. If my library disappears tomorrow, opp-capture keeps working. Clean boundaries, clean mind.

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

The model outputs one flag name per line, nothing else. For example:

```
waterfall_methodology
no_remote_work
state_specific_requirement
```

This is the simplest possible format for a small model to follow reliably. Flag names are looked up against the static taxonomy in application code — tier, definition, and any other metadata come from there, not from the model. The application layer constructs the final structured response, so structured output is guaranteed regardless of model behavior. Parsing is `output.strip().splitlines()` filtered to known flag names.

The system prompt enumerates all valid flag names explicitly and instructs the model to draw only from that list. The flag set assigned to the small model is small enough (~18 flags) that full enumeration is trivial token cost. This eliminates hallucinated flag names at the source rather than relying solely on post-processing. The RAG serves context enrichment (definitions, examples), not as the gating mechanism for what names are valid — constraining output to RAG results would turn retrieval misses into false negatives.

### Architectural Boundary

F4 is a Python library in its own repo (`f4-plugin`) with its own infrastructure code (Terraform for Bedrock Custom Model Import, IAM roles). Opp-capture imports F4 and calls `f4.filter(text)`, which returns a dict with detected flags, a FILTER/REVIEW decision, and processing metadata. The pipeline logic (chunking, RAG, inference, parsing, deduplication, decision) runs in-process — F4 is not a deployed web service.

On the opp-capture side, the integration surface is minimal: a new adapter behind an existing port, plus a config toggle. Opp-capture may want to disable F4 at times to avoid costs when the pipeline doesn't need it. If F4 gets deleted, opp-capture loses nothing. This follows opp-capture's own OEA principles — no invasive species.

Consequences: I own my own infra (Bedrock model import, IAM) but not running compute. Clean separation. The opp-capture adapter is minimal — a new adapter behind an existing port — and is part of the evaluation plan for delta comparison.

### Library Contract

Thick contract. `f4.filter(text)` accepts RFP text and returns a dict containing detected flags, a FILTER/REVIEW decision, and processing metadata. The flow inside the library: chunk the text → send chunks to the model concurrently (bounded concurrency) → parse and retry once on failure → deduplicate flags across chunks → apply algorithmic decision logic → return result.

The response includes an `unparsed_chunks` count so the caller knows if any (and how many) chunks failed parsing after retry.

### Chunking Strategy

RFPs will be chunked at roughly 512-1024 tokens as a starting point, tunable based on testing. Chunks will overlap to avoid missing context at boundaries. Each chunk is independently sent to the model for flag detection. Flags are deduplicated across all chunks — either a flag is present in the RFP or it isn't.

Known limitation: some flags (e.g., `brownfield`, `feature_factory`, marginal investment sub-criteria) may only be inferable from context spread across multiple chunks. The independent-chunk strategy may miss these. This is acknowledged and deferred — not addressed in this project, but a candidate for post-final improvement if evaluation shows it's a meaningful problem in practice.

Chunks are sent to the Bedrock endpoint concurrently using a bounded concurrency pool (tunable max_workers, default TBD based on testing). Bedrock Custom Model Import endpoints handle concurrent invocations, but have throttling limits that depend on provisioned CMUs. Wall-clock inference time scales with chunk count divided by concurrency limit rather than chunk count alone. This is a free latency win — Bedrock pricing is per-CMU-minute regardless of utilization, so parallel requests should cost the same as sequential ones.

Per-chunk retry: if the model output fails to parse, retry that chunk once. If it still fails, discard the chunk and increment the `unparsed_chunks` counter. The full RFP is not retried. With high per-attempt compliance, the probability of failing both attempts on a single chunk is low — discarded chunks stay acceptable noise rather than meaningful signal loss. This only holds if compliance is actually high; fine-tuning evaluation must confirm it.

### Flag Taxonomy

Collated from my previous biz-dev work and opp-capture's existing evaluation config. Four severity tiers: fast-fail, red, green, and blue (informational). Full flag set documented in `collated-flag-set.md`. The flag set is in draft and requires manual review for finalization.

### Decision Logic

TBD. The algorithmic rules mapping flag combinations to FILTER/REVIEW outcomes are configurable and can be adjusted in code without retraining. This is a business policy conversation that may happen after the final presentation, in collaboration with biz-dev. The configurability is a feature — different organizations could define their own flag sets and decision rules.

### Model Selection

**Selected: meta-llama/Llama-3.2-3B-Instruct**

All candidate architectures (Llama, Qwen, Mistral) are supported by Bedrock Custom Model Import. Model selection was driven by IFEval score — the primary proxy for instruction-following and output format compliance, which is the highest risk given no constrained generation on Bedrock.

IFEval scores from the Open LLM Leaderboard:

- Llama 3.2 3B-Instruct: **73.93%**
- Qwen2.5-3B-Instruct: 64.75%
- Mistral-7B-Instruct-v0.3: 54.65% (7B, no 3B variant available)
- Llama 3.2 1B-Instruct: 56.98% (ruled out on IFEval and capacity)

Llama 3.2 3B also has a proven Bedrock Custom Model Import path — the same architecture was successfully imported in a prior assignment (1B variant), including a known fix for the `tokenizer_class` field in `tokenizer_config.json` (must be the user-facing class name, e.g. `LlamaTokenizerFast`, not the backend class).

Fallback if 3B underperforms after fine-tuning: Llama 3.1 8B-Instruct or Qwen2.5-7B-Instruct.

After LoRA fine-tuning, adapters must be merged back into the base model and exported as Hugging Face safetensors format for Bedrock Custom Model Import.

### Datasets

Data Sources:
- Synthetic chunks: generated by Claude (Sonnet 4.6) via distillation — the small model learns to replicate the large model's judgment on flag detection. RFP-style text chunks labeled with correct flags, including examples with no flags, single flags, multiple flags, adversarial examples (text that resembles a flag but isn't), and ambiguous text. Target volume: ~2,500 examples (~100 per flag), generated via a script calling the Claude API.
- Real RFP chunks: extracted from actual RFPs, manually labeled with expected flags. Tests generalization beyond synthetic data — different writing styles, different flag distributions. This is the test set.

Train/Eval/Test Split:
- Train Set Data: 80% of Synthetic Set.
- Eval Set: 20% of Synthetic Set
- Test Set: Small set of real RFP chunks.

### Evaluation Strategy

Finetuning evaluation across training epochs and on the held-out eval set will depend on three metrics.
- Flag Level Precision: Of the flags the model detected on a chunk, how many were actually present? Low precision on a fast-fail flag kills a good opportunity (missed potential revenue).
- Chunk Level Recall: Of the flags actually present, how many did the model detect? Low recall lets a bad RFP through (wasted effort).
- Format Compliance: Of all chunks, what fraction produced parseable output with only valid flag names? This is a first-class metric — parse failures inflate `unparsed_chunks` and silently degrade recall. Compliance rate should improve across fine-tuning epochs; if it doesn't, retry is masking a worse underlying problem.

An additional evaluation will occur after plugging into the opp-capture pipeline.
- Delta against current filtering: run both F4 and opp-capture's existing string/metadata-based filtering on the same set of RFPs during a production run, and compare. Did F4 catch flags that string matching missed? Did the current system get false-positives that F4 parsed correctly?

### RAG Purpose

ChromaDB vector store containing flag definitions, example passages, and real RFP language. The RAG serves four purposes:

1. Patch gaps: if training missed certain flag expressions, add example passages post-training without retraining
2. Extensibility: add entirely new flag types via definitions and examples in the vector store. The model receives these as in-context examples at inference time and can detect the new flags without retraining. Retraining can happen later if enough new flags accumulate. This capability may depend on model selection — a 7B model is much more likely to generalize to unseen flag IDs via in-context learning than a 3B model. Validate this when the finetuned model is performing well.
3. Evaluation: measure whether RAG-augmented finetuned inference outperforms base fine-tuned inference (stretch goal)
4. Contect Enrichment: The meta explanation of a flag, and real RFP language in the RAG, gives the model context that synthetic training data may miss

At inference time, each chunk is embedded and the top-k most similar flag definitions and example passages are retrieved from ChromaDB and prepended to the prompt as context. The value of k is tunable — too few and relevant flags are missed, too many and the prompt is diluted with noise and inflated context size (on top of the chunk's own 512-1024 tokens).

### Prompt Optimization

TextGrad, applied after fine-tuning but before Bedrock deployment. TextGrad treats the system prompt as a trainable variable and uses an LLM backward engine to generate critiques and rewrites. The forward engine is the fine-tuned model loaded locally on SageMaker; the backward engine is Claude Opus via Bedrock (SageMaker can call Bedrock with the right IAM role). Running the forward engine locally avoids CMU billing and cold starts during the many forward passes TextGrad requires per optimization step. The optimized prompt is then used when the model is deployed to Bedrock.

If it doesn't help, I will document that, and at least I tried.

### Frontend

Gradio app with a `gr.File()` upload component. Accepts PDF/DOCX, extracts text (duplicating opp-capture's extractors for demo convenience), calls `f4.filter()`, and displays the flag report and decision. This is a demo surface — the real integration point is the library.

For the demo, the app runs locally and is exposed via Gradio's built-in `share=True` tunnel, which generates a temporary public URL (expires after 72 hours). Password-protected via Gradio's `auth` parameter so only the attendees can access it. Bedrock calls originate from the local machine using Flexion's AWS credentials — shut down the process immediately after the demo to stop exposing the endpoint.

Consequences: text extraction is duplicated from opp-capture. The library itself accepts pre-extracted text. For production use via opp-capture, text extraction happens on their side.

### Text Input

`f4.filter()` accepts pre-extracted text, not files. Opp-capture already handles text extraction (pdfplumber, docx). The Gradio frontend handles extraction locally for demo purposes only.

### Observability

The library will log per-batch running time, token counts (from Bedrock response metadata where available, estimated client-side via the model's tokenizer otherwise), and inference cost. This is essential — without it, there's no way to empirically compare F4 against the Claude baseline (see below) and justify or invalidate the fine-tuned model approach.

### Cost vs. Claude Baseline

The obvious alternative is skipping the fine-tuned model entirely and sending each RFP to Claude (already available via Bedrock in opp-capture) with a flag-detection prompt. This would be simpler — no training data, no fine-tuning, no chunking, no RAG, and Claude's large context window handles full documents without chunking. The cost case for F4 over Claude depends on batch volume, document size, and how many sources opp-capture queries. At low volume, Claude is likely cheaper and simpler. At high volume, F4's time-based pricing amortizes better than Claude's per-token pricing, especially for large documents.

The latency case is less obvious. Chunking introduces more inference calls than Claude's single-document approach. However, F4 sends chunks concurrently with bounded parallelism (see Chunking Strategy), so wall-clock time scales well below linear with chunk count. Opp-capture's Analyzer also makes two sequential Claude calls per opportunity (summarize then evaluate), so F4's parallel chunks may actually be faster despite the smaller model processing more requests.

This is one reason F4 tracks running time and token usage per batch (see Observability above) — so the comparison can be made with real numbers, not estimates.

## Consequences

This project touches nearly every skill from the course: LoRA fine-tuning, RAG, prompt optimization, constrained output via prompting, evaluation, Bedrock deployment, and frontend. The risk is breadth — there's a lot to build in two weeks.

The configurable flag set and decision logic make this potentially productizable beyond Flexion. There's no reason another organization couldn't define their own flags and use this library for their own RFP screening.

The standalone architecture means I can develop and demo independently of opp-capture. The opp-capture adapter is needed for the delta evaluation but is minimal work given opp-capture's existing port/adapter pattern.

## Rubric Alignment

Model & Inference: LoRA fine-tuned model, synthetic training data, real RFP test set, RAG for extensibility, TextGrad prompt optimization.

Production Environment: Bedrock Custom Model Import. Library with own infra (Terraform for model import, IAM). Opp-capture adapter for delta evaluation. Observability for cost comparison against Claude baseline.

Inference Pipeline: `f4.filter(text)` → chunking → RAG retrieval → concurrent model inference → parse/retry → flag deduplication → algorithmic decision → result dict.

Documentation: This ADR, plus technical docs on model selection, training data generation, evaluation results, and deployment.

Demo: Gradio frontend — upload an RFP, get a flag report and decision. Delta comparison against current string/metadata filtering if possible.

## Notes

F4 = Flexion Fast Fail Filtering.

Document edited with LLM aid.
