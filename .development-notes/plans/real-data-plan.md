# Real RFP Data Labeling Plan

## Goal

Replace synthetic training data with Claude-labeled real RFP data. The model works end-to-end
on Bedrock but detects zero flags on real RFPs — the domain gap between synthetic and real text
is the bottleneck.

## Data Source

`/Users/travisblount-elliott/library 2/downloads/` — 958 RFP directories, 2,691 files (mostly
PDF/DOCX), 2.1 GB total. These are real government RFP solicitations downloaded by opp-capture.

## Target Volume

~3,000 labeled chunks total, split as:
- RAG seeds: ~200 (best examples, ~10 per flag)
- Training: ~2,240 (80% of remaining)
- Eval: ~280 (10% of remaining)
- Test: ~280 (10% of remaining)

This is comparable to the synthetic dataset (2,138 examples) but with real text.

## Approach: Claude Extracts + Quota Tracking

Instead of mechanically pre-chunking and labeling each chunk, Claude reads large sections of each
RFP and extracts flag-relevant passages itself. This is more efficient — Claude skips boilerplate,
finds the best chunk boundaries around flag signals, and produces higher-quality training data.

### Per-Flag Quotas

Track a running count per flag. Target ~150 examples per flag (19 flags × 150 = 2,850 flagged
chunks). Additionally, target ~150 no-flag chunks for a total of ~3,000.

Once a flag hits its quota, tell Claude to stop extracting for that flag (include the filled
flags in the prompt). This forces distribution across all 19 flags and avoids wasting API calls
on common flags like set-asides once they're saturated.

The script stops when all quotas are filled, or when all RFPs are exhausted.

### Sampling Strategy

1. Randomly shuffle all 958 RFP directories
2. Process RFPs one at a time:
   a. Extract text from all PDF/DOCX files in the directory
   b. Combine into a single text block
   c. If text exceeds ~30,000 words, split into sections of that size
   d. Send each section to Claude with extraction instructions + current quota status
   e. Claude returns extracted passages with labels
   f. Append to output, update per-flag counts
   g. Skip to next RFP if all quotas for this RFP's likely flags are full
3. Stop when all quotas filled or RFPs exhausted

### Claude Extraction Prompt

For each RFP section, Claude (Sonnet via Bedrock) receives:
- Flag definitions and descriptions (all 19 flags)
- Which flags still need examples (quota status)
- The RFP text section
- Instructions to:
  - Find passages (~200-400 words) that exhibit any of the needed flags
  - Also extract ~1-2 passages that clearly contain NO flags (for no-flag training)
  - For each extracted passage, output the flags, confidence, and adversarial rating
  - Extract the best chunk boundaries — the passage should be self-contained and contain
    clear evidence for the flag

Example Claude output format (JSON array for easy parsing):
```json
[
  {
    "chunk_text": "This solicitation is set-aside for SDVOSB businesses...",
    "flags": ["sdvosb_set_aside"],
    "confidence": "high",
    "adversarial": false
  },
  {
    "chunk_text": "The contractor shall provide IT modernization services...",
    "flags": [],
    "confidence": "high",
    "adversarial": false
  }
]
```

If no flags from the needed list are found in the section, Claude returns an empty array or
only no-flag passages.

Claude labels become the ground truth for training. The confidence and adversarial fields are
used for RAG seed selection, not for training labels.

## RAG Seed Selection

After all chunks are labeled:
1. Filter to high-confidence, non-adversarial chunks that have at least one flag
2. Group by flag
3. For each flag, take the top examples (aim for ~10 per flag)
4. Half go to RAG seeds, half stay in training pool
5. This ensures RAG seeds and training examples are from different specific chunks

RAG seeds are written to `data/rag_seeds_real.jsonl` (same format as existing `rag_seeds.jsonl`).
At inference, both synthetic and real seeds populate ChromaDB.

## Output Files

The script produces a single raw output file first:
```
data/labeled_real.jsonl  — all labeled chunks with metadata
```

Each line:
```json
{
  "rfp_id": "00b8b581c63f44389c7ef5fe55b6638c",
  "chunk_index": 0,
  "chunk_text": "...",
  "flags": ["sdvosb_set_aside"],
  "confidence": "high",
  "adversarial": false
}
```

A second script (or second phase of the same script) converts this to training format:
1. Select RAG seeds from high-confidence chunks
2. Build ChromaDB from combined synthetic + real RAG seeds
3. For each remaining chunk, query ChromaDB for RAG context (excluding self)
4. Format as training JSONL: system prompt + `[Retrieved context]...` + chunk → flags
5. Shuffle and split into train/eval/test
6. Write to `data/train.jsonl`, `data/eval.jsonl`, `data/test.jsonl`

## Data File Reorganization

Before running the labeling script, reorganize existing files:

```
data/
  synthetic/              ← archive current synthetic data
    train.jsonl           ← mv from data/train.jsonl
    eval.jsonl            ← mv from data/eval.jsonl
    test.jsonl            ← mv from data/test.jsonl
  backup_v1/              ← delete (pre-rewrap synthetic, no longer needed)
  rag_seeds.jsonl         ← keep (synthetic seeds, still useful alongside real seeds)
  rag_seeds_real.jsonl    ← new (real RFP seeds from labeling)
  labeled_real.jsonl      ← new (raw Claude labels before split)
  train.jsonl             ← new (real data, active training set)
  eval.jsonl              ← new (real data, active eval set)
  test.jsonl              ← new (real data, active test set)
```

## Script: `scripts/label_real_data.py`

```
Usage:
  PYTHONPATH=. uv run python scripts/label_real_data.py --source-dir "/path/to/downloads"
  PYTHONPATH=. uv run python scripts/label_real_data.py --source-dir "/path/to/downloads" --target-per-flag 10 --dry-run
  PYTHONPATH=. uv run python scripts/label_real_data.py --source-dir "/path/to/downloads" --target-per-flag 150 --seed 42
```

Arguments:
- `--source-dir`: path to RFP downloads (required)
- `--target-per-flag`: target examples per flag (default: 150, total = 19 flags × 150 + 150 no-flag)
- `--target-no-flag`: target no-flag examples (default: 150)
- `--output`: output file (default: `data/labeled_real.jsonl`)
- `--seed`: random seed for reproducible sampling
- `--dry-run`: extract text but don't call Claude, just report extraction stats
- `--resume`: append to existing output file, skip already-processed RFP IDs

Key behaviors:
- Tracks per-flag quotas, passes filled flags to Claude so it skips them
- Resumes from partial runs (writes incrementally, tracks processed RFP IDs)
- Reports progress: "Processed 50/958 RFPs | waterfall: 45/150, sdvosb: 150/150 (full), ..."
- Canary: after 30% of RFPs processed, prints full quota status every 10 RFPs
  so you can see which flags are filling and which are rare
- Skips RFPs where all likely flags are already at quota
- Handles extraction failures gracefully (skip, log, continue)
- Stops when all quotas filled or all RFPs exhausted

## Script: `scripts/build_training_set.py`

Converts `labeled_real.jsonl` into the final training files.

```
Usage:
  PYTHONPATH=. uv run python scripts/build_training_set.py
  PYTHONPATH=. uv run python scripts/build_training_set.py --rag-per-flag 10 --split 80/10/10
```

Steps:
1. Load `labeled_real.jsonl`
2. Select RAG seeds (high confidence, non-adversarial, ~10 per flag, split half to RAG / half to training)
3. Write `data/rag_seeds_real.jsonl`
4. Build ChromaDB from synthetic + real seeds
5. For each remaining chunk, wrap with RAG context (excluding self from results)
6. Format as training JSONL
7. Shuffle and split
8. Write train/eval/test

## Test Run

Before the full run, do a test with `--target-per-flag 10`:
1. Verify extraction works across PDF/DOCX
2. Verify Claude labeling returns valid JSON
3. Verify confidence/adversarial fields populated
4. Verify output format matches expected schema
5. Manually inspect a few labels for correctness

## Questions to Resolve Before Execution

1. Should we mix synthetic + real data in training, or go pure real? Starting with pure real
   isolates the variable. Can always mix later if recall drops on certain flags that real data
   lacks examples for.
2. What if certain flags are very rare in real RFPs (e.g., design_exercise, oral_presentation)?
   The quota system will keep processing RFPs until quota is met. If 958 RFPs are exhausted
   before a flag hits 150, supplement with targeted synthetic examples for that flag.
3. Should the labeling prompt include RAG context (like inference) or just the raw chunk?
   Just the raw RFP text — Claude doesn't need RAG help. RAG context is added later when
   building the training set via `build_training_set.py`.
