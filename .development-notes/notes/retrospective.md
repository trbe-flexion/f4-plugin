# F4 Retrospective — What Needs to Change

## 1. Data Gathering

Data quality has been the persistent bottleneck. The current approach — batch-labeling RFP chunks with Claude after the fact — produced a dataset that required multiple rounds of validation, relabeling, and flag pruning. Even after Opus validation caught a ~13% error rate in Sonnet labels, the resulting training set was small (493 examples) and skewed toward common flags.

**What to do differently:** Integrate data gathering directly into the opp-capture pipeline. The pipeline already runs a large model (Claude Sonnet) to evaluate whole RFPs. The change:

1. Inject F4's chunking step and flagging prompt into the live pipeline as a passive side-channel — chunk the RFP text and send each chunk through the flag-detection prompt alongside the existing evaluation.
2. Focus on 1–2 flags at a time. This keeps the labeling prompt tight and the per-chunk cost low.
3. Use the existing whole-RFP evaluation output as a cross-reference. If the large model's document-level assessment says "no set-aside requirements" but a chunk gets labeled `small_business_set_aside`, that's a signal to review. This gives a built-in quality check that the batch-labeling approach lacked.
4. Spread the cost over time. Running continuously against real traffic means the dataset grows organically without a large one-time labeling bill.

Once a baseline of per-flag data exists, prompt engineering becomes viable on a flag-by-flag basis — tuning the detection prompt for each flag's specific patterns and edge cases, informed by real examples rather than guesswork.

The result: a steadily growing, high-quality, real-world dataset with built-in cross-validation — instead of a one-shot synthetic/batch-labeled corpus that required extensive cleanup. And a natural on-ramp to per-flag prompt optimization once there's enough data to work with.

## 2. Specialist Models Instead of One Generalist

We started with 30+ flags and had to cut down to 7 through iterative evaluation. A single 3B model trying to detect that many flags at once was spreading its capacity too thin. Flags that are easy to detect (like `oral_presentation`) compete for model attention with flags that are harder or more ambiguous, and similar flags (the set-aside family) can confuse each other. Worse, the black flags we dropped all struggled with the same problem — they need document-level context that chunk-level detection can't provide — but training the model to be strict about those flags risked suppressing detection on the easier ones.

**Proposed strategy: 2–3 specialist models grouped by flag family and detection strategy.**

- **Set-aside model:** Trained only on `small_business_set_aside`, `8a_set_aside`, `sdvosb_set_aside`, `hubzone_set_aside`. These flags share similar RFP language (contracting/eligibility sections) and the model can learn finer distinctions within that family without being distracted by unrelated flags.
- **Marginal-reason model:** Flags like `marginal_cost`, `marginal_schedule`, etc. — a different detection pattern focused on evaluation criteria language.
- **Black-flag model:** The document-level context flags (`waterfall_methodology`, `no_custom_development`, etc.) that were dropped from the current model. A specialist could be trained to be *strict* about interpreting any flags in this category without that strictness bleeding over into easier detection tasks.

**Gated evaluation:** For example: the black-flag model could run first. If it detects anything, stop — don't bother running the other models. This turns the cheapest, fastest check (does this RFP have a disqualifying characteristic?) into a gate that avoids unnecessary inference on opportunities that are already dead.

## 3. RAG Needs Specialization Too

RAG was built, wired in, and retrained with — and it made things worse (F1 dropped from 94.1% to 83.0%). For a generalist model detecting 30+ flags, the retrieved context was too noisy. Flag language varies enormously across flag families, so retrieving "relevant" examples for a chunk meant pulling in examples from multiple unrelated flags, diluting the signal.

With specialist models, RAG becomes viable again. A set-aside model's RAG store would only contain set-aside examples — tightly scoped retrieval with much less noise. The context budget isn't split across unrelated flag families, so each retrieved example is actually informative.

## 4. Chunk Size Was Too Conservative

The 512-token chunk size was chosen early, when RAG context was expected to consume a significant portion of the model's context window alongside the chunk and system prompt. With RAG disabled, that budget is no longer needed — but the chunk size was never revisited. 512 tokens is a small window for detecting flags that depend on surrounding context, and it's likely a contributor to the recall problems on flags that were eventually dropped.

This is worth revisiting now. The Bedrock Custom Model Import may also impose a context limit smaller than the base model's native window — that hasn't been fully characterized yet. But even within known limits, there's almost certainly room to send larger chunks and improve recall, especially for flags where the signal is spread across a few sentences rather than concentrated in a single clause.

The tradeoff isn't purely "bigger is better." Larger chunks give the model more context, but they also increase the chance of including unrelated content that could trigger false positives. The right chunk size likely depends on the flag family — another argument for specialist models with tuned chunk sizes per task.

Chunk size has major downstream impact — it affects what the model can see, what flags are even detectable, how many inference calls per RFP, and how training data is structured. A conservative early estimate rippled through the entire pipeline.
