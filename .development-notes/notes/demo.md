# F4 Demo (10 minutes MAXIMUM)

---

## 1. Context: Opp-Capture Today (~1 minute)

Flexion has an existing pipeline called opp-capture that screens government RFPs for business fit. It has four stages — Scout, Screener, Downloader, and Analyzer. The Screener uses string-based keyword matching before passing an RFP to the Analyzer. Keyword matching is fast, but has obvious limitations. The Analyzer uses Claude Sonnet via Bedrock for deeper evaluation. It's good, but relatively expensive, slow, and isn't perfect. The gap between dumb keyword matching and a full LLM evaluation is where we lose opportunities or waste biz-dev's time. I think a small specialist model could help.

---

## 2. The Idea: A Small Model in the Gap (~3.5 minute)

F4 stands for Flexion Fast Fail Filtering. It's a library that plugs into opp-capture, and uses a fine-tuned 3B parameter model that detects specific flags in RFP text — faster and cheaper than Claude, smarter than string matching, and possibly more accurate than Sonnet for some flags.

### What it does

The library exposes a single filter function that accepts text. It chunks the RFP, sends chunks concurrently to the Bedrock custom model for inference, parses the flags out of each response, deduplicates across chunks, and runs an algorithmic decision to produce a FILTER or REVIEW recommendation. It's a standalone Python library — opp-capture can import it with an adapter behind an existing port. Just keeping the boundaries clean.

### Model Decision

The constraints shaped the model choice.

* I needed small, fast, and cheap, but not stupid model - based on experience in the class and metrics on HuggingFace - 3B felt like the sweet spot of meaningfully cheap and capable.
* Bedrock doesn't support constrained generation for custom models — no grammar enforcement, no JSON mode except best-effort. This meant I needed a model with a high IFEval score to ensure output compliance.
* The model obviously also needed to be supported by Bedrock Custom Model import, which only supports a handful of architectures.

Llama 3.2 3B-Instruct was the best fit among my constrained options. IFEval: ~74% OOTB and this was the deciding factor.

### Design Choices

More about why Bedrock - opp-capture already interfaces with it, and we're already in the AWS ecosystem. Custom Model Import also gives me features I felt were important for a bursty-use case. It scales to zero after idle, and pricing is time-based rather than per-token. No GPU instances to manage.

To mitigate the format risk, I made the output format as simple as possible — one flag name per line, nothing else. This routinely achieved over 99% compliance after fine-tuning.

The application layer handles everything structural: taxonomy lookup, tier assignment, decision logic. This gives us a lot of levers to adjust based on raw output.

I chose to chunk text to 512 tokens.
* I thought too large might be noisy.
* I believed I would be sharing context with RAG results.
* I believed I could run into context window problems with larger chunks.
* I wanted my prompts to be reasonably human-readabnle for sanity-testing.

### Flag taxonomy

I started with over 30 flags.
After more than a dozen rounds of training and testing, I reduced that down to seven flags.
The ones I kept are the ones the model could reliably detect at the chunk level.

The flags I dropped fell generally into two categories:
* They are rare! Not enough training data even with 900 RFPs to draw from!
* They required more context than a 512-token chunk could provide (like whole-document level)

---

## 3. Live Demo (~1 min)

This is a Gradio frontend test harness. I can upload a PDF or DOCX, and F4 processes it end-to-end. What you'll see is the pipeline hitting the Bedrock endpoint for real.

*[Upload test RFP, walk through output]*

**Pre-demo:**
```
uv run python -m src.frontend --model-arn "arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:imported-model/3ffr95d8c4cc" --share --auth demo:demo
```

---

## Process, Reflections, the Future

### Iteration Hell (1 minute)

I ran many train-eval-test cycles. The first POC to get me to a full slice used synthetic data to train, but didn't work well with real RFPs (expected).

I was given 950+ real RFPs by Tom Willis, so I set about cleaning them with a script running them through Sonnet to isolate chunks that would have associated flags. That worked poorer than I expected.

From there it was iterative data cleaning. Every major improvement came from data quality. Adjusting hyperparameters either yielded no meaningful improvement or a regression. F1 climbed from 39% to 65% mostly through data cleaning and overall scope reduction.

I identified several fag detection patterns that seemed to be causing tension in the model, and eventually settled on a set of only 7 flags that performed well and did not cause detection tension with each other. With this set, I achieved 92% precision and 85% recall.

### Take-aways (1.5 minute)

Data Quality: An Opus validation pass caught a 13% error rate in the training data. Prompt engineering for the *labeling step* of distillation was just as important as prompt engineering for fine-tuning and inferrence itself. Conducting a series of experiments, I found I was able to produce very high quality ground truth data, but I estimated it would cost at least $1000 dollars to process my 950+ RFP set - so I did not do that.

A 3B model struggled when the fine-tuning patterns were different. Example! Flags like scope alignment or brownfield struggled with a 512 token chunk - they needed to infer whether the chunk was part of a summary, or an offhand or out-of-context mention of related concepts. Flags like waterfall development were practically looking for a keyword and and asking if it was a positive or negative, so document-level context inferrence was detrimental. Training it to infer document-level context for some flags but explicitly ignore context for others caused either or both types of detection to fail catastrophically.

RAG was Detrimental: Retrieved examples from 30+ flag types were noisy enough to degrade performance rather than help.

[If I have time: Integration is always hard. The model worked perfectly on SageMaker but produced gibberish on Bedrock for any prompt longer than about 875 tokens. I binary-searched the exact token cutoff and eventually traced it to a field name in config.json. The export script used `rope_parameters`, which is the transformers 5.x field name, but Bedrock's inference container expects `rope_scaling`, the 4.x name. Without RoPE scaling, the position encoding broke at longer sequences, abruptly at exactly 876 tokens. Renaming the field fixed it.]

### How I will proceed (2.5 minutes)

Chunk length: Revisit this. It's worth finding the balance between enough context and creating too much noise for a small model, for this task, experimentally.

Data: My approach to gathering clean data was sloppy in retrospect. Distillation was a good approach, but instead of asking a big model to process a whole RFP and extract relevant chunks, it would be better to chunk-then-infer, providing the larger model plenty of prompt context. I experimented with a dual-pass method where the second prompt would grade the first. This produced VERY high quality data, but as I said, would have been too expensive.

Potential Data Solutions:

* Create a data-cleaner and integrate it into the opp-capture pipeline for a period of time. Each RFP that passes through would get chunked, evaluated, and distilled into high quality labeled chunks. Cost spreads over time.
* Run the dual-pass method script over just a few historical RFPS each day/week.
* Data quality may also be improved by labeling for only one type, detection strategy, or family of flag at a time - a much more focused task than trying to label over a dozen flags at once.

Prompts: A starter high quality set would also allow me to tune prompts for training or detection.

Multiple Fine-Tuned Models: I could train multiple small models to focus on flags with related detection strategies. For instance, many of the black flags naturally group by detection strategy.

Gating: The multiple models approach could be gated to prevent unnecessary inferrence - most RFPs should fast fail! Drawback: MORE inferrence passes on modderate to good fit RFPs.

Rag: A RAG may become viable again with specialist models.

---

## Prep checklist

- [ ] Warm up Bedrock endpoint ~5 min before (run smoke test)
- [ ] Have a test RFP ready that triggers 2-3 flags
- [ ] Test `--share` tunnel from presentation machine
