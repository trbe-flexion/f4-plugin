# Validation Pass Plan

## Goal

Use Claude Opus to review all Sonnet-labeled chunks in `data/labeled_real.jsonl` and cull
mislabeled or ambiguous training examples. Spot checks showed ~13% error rate concentrated
in specific flags (no_custom_development, large_team, marginal_short_duration, waterfall,
brownfield).

## Design

**Script:** `scripts/validate_labels.py`

### Input/Output
- Input: `data/labeled_real.jsonl` (Sonnet labels)
- Output: `data/validated_real.jsonl` (Opus-reviewed, same schema + `validation` field)

### Batching
- 10 chunks per Opus call (~3-4k words per batch)
- ~4000 chunks / 10 = ~400 API calls
- Each call asks Opus to review Sonnet's labels and verdict each chunk

### Concurrency
- `asyncio` + `anthropic.AsyncAnthropicBedrock`
- Semaphore of 5 concurrent requests
- Retry with exponential backoff on throttling (429)

### Opus Prompt Per Batch
For each batch of 10 chunks, Opus receives:
- Flag definitions (same 19 flags)
- The 10 chunks with Sonnet's labels
- Instructions:
  - For each chunk, output one of:
    - `KEEP` — labels are correct as-is
    - `FIX` — provide corrected flag list (may add unambiguous missing flags or remove wrong ones)
    - `DROP` — chunk is too noisy/ambiguous to be useful training data
  - Be conservative with FIX-add: only add a flag if the signal is unambiguous
  - Be aggressive with DROP: borderline examples hurt more than help in LoRA training

### Output Format
Same schema as input, plus:
```json
{
  "rfp_id": "...",
  "chunk_index": 0,
  "chunk_text": "...",
  "flags": ["corrected_flags"],
  "confidence": "high",
  "adversarial": false,
  "validation": "keep|fixed|dropped",
  "original_flags": ["sonnet_original_flags"]
}
```

### Resume Support
Track processed chunk indices via output file. `--resume` skips already-validated chunks.

### Stats
Print summary at end:
- Total kept / fixed / dropped
- Per-flag: added, removed, net change
- Error-prone flags breakdown

## Estimated Cost
- ~400 Opus calls × ~4k input tokens × ~500 output tokens
- Input: ~1.6M tokens × $15/MT = ~$24
- Output: ~200k tokens × $75/MT = ~$15
- Total: ~$39

## After Validation
1. Filter `validated_real.jsonl` to exclude `dropped` rows
2. Feed into `build_training_set.py` (uses validated flags, not originals)
3. Supplement rare flags with seeded synthetic
4. Retrain
