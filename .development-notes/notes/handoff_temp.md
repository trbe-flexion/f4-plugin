# Handoff: Relabeling Pipeline — In Progress

## What's Running

`scripts/relabel_training_data.py` is running the full relabeling pipeline across all 958
source RFPs. This generates new training data for the fine-tuned Llama 3.2 3B model.

```bash
# Phase 1 (may already be done — check if relabeled_pass1.jsonl exists and has data)
PYTHONPATH=. uv run python scripts/relabel_training_data.py --phase 1 --concurrency 10

# Phase 2 (run after Phase 1 completes)
PYTHONPATH=. uv run python scripts/relabel_training_data.py --phase 2 --concurrency 10

# Or both sequentially:
PYTHONPATH=. uv run python scripts/relabel_training_data.py --phase both --concurrency 10
```

Both phases have resume support — safe to kill and restart. Phase 1 resumes by
(rfp_id, chunk_index) key. Phase 2 resumes by record position.

## What Changed This Session

We spent this session iterating on the labeling prompts. Major changes from the
prior handoff:

### Script changes (`scripts/relabel_training_data.py`)

1. **`--max-rfps` now limits extraction** — Previously, all 958 RFPs were extracted
   and chunked before any filtering. Now `--max-rfps N` slices the directory list
   before extraction, so `--max-rfps 10 --max-flagged 20` is fast for spot-checks.

2. **Dropped `hardware_procurement` flag** — Removed from `KEEP_FLAGS` and
   `FLAG_DESCRIPTIONS`. Redundant with `scope_misalignment` (if the RFP is for
   hardware, scope_misalignment already covers it). Down to 18 flags.

3. **Rewrote `scope_misalignment` description** — Old version triggered on any
   chunk describing non-software work. New version requires the chunk to be a
   top-level scope statement (intro, overview, purpose section). Task-level SOW
   details, work breakdown items, and operational procedures no longer trigger it,
   even if they describe non-software work. This was the biggest source of FPs.

4. **Rewrote `off_the_shelf_software` description** — Old version used keyword
   matching (COTS, GOTS, etc.). New version uses the same conceptual pattern as
   scope: chunk must explicitly state what the procurement is for, and the primary
   deliverable is acquiring/licensing/standing up a commercial product. Added
   explicit exclusion: ongoing support/maintenance/operations of commercial products
   already in use does not qualify.

5. **Added high-confidence warning to all black flags** — `scope_misalignment`,
   `off_the_shelf_software`, `budget_too_low`, `onsite_required` all end with:
   "A false positive permanently discards a revenue opportunity — only flag when
   confidence is very high."

6. **Rewrote Phase 2 validation prompt** — Old version was too deferential (0/20
   fixes on first test). New version:
   - Leads with "CRITICAL: Return ONLY a JSON object" to prevent reasoning preamble
   - Uses terse removal criteria instead of adversarial framing (which triggered
     chain-of-thought that blew past max_tokens)
   - BLACK_FLAGS set marks black flags with `[BLACK — fast-fail]` prefix
   - Explicit admin section exclusion (eval criteria, CLINs, SF 1449, FAR provisions)
   - Now achieves meaningful fix rates (~20-40% of flagged chunks get at least one
     flag removed)

7. **JSON parsing hardened** — Extracts JSON from `first { to last }` in response
   to handle cases where model prepends reasoning text. Debug logging added for
   parse failures (prints first 200 chars of raw response).

8. **Progress reporting** — Every 100 chunks now shows `X/Y RFPs` alongside chunk
   and flag counts.

### Prompt doc (`test-labeling-prompt.md`)

Updated to match all script changes. This is the source of truth for prompt text —
always update the prompt doc FIRST, then update the script to match.

Note about prefill: Bedrock custom model import does NOT support assistant message
prefill. We tried and got 400 errors. JSON compliance is enforced via prompt
instructions only.

## Validation Results (Small Sample)

Final test run: 5 RFPs, `--max-flagged 40`, both phases.

**Phase 1**: 86 chunks, 23 flagged
**Phase 2**: 18 flagged (8 fixes — removed 6 scope, 3 OTSS)

**0 false positives** across 18 flagged chunks in the final output. Flags that fired:
scope_misalignment (9), oral_presentation (4), small_business_set_aside (2),
lpta_source_selection (2), design_exercise (2), off_the_shelf_software (1),
sdvosb_set_aside (1), onsite_required (1), marginal_short_duration (1).

Flags that have NOT been tested yet (no examples in 5-RFP sample):
waterfall_methodology, 8a_set_aside, wosb_set_aside, hubzone_set_aside,
agile_methodology, budget_too_low, onsite_madison, brownfield, large_team.

## Open Question: Keep Black Flags?

We discussed whether to keep the judgment-heavy black flags (scope_misalignment,
off_the_shelf_software) in training or drop them and let the downstream LLM handle
them. Decision was deferred — "one more test call" at scale. The full run will
answer this: if black flag FP rate stays under ~5% across 958 RFPs, keep them.

Middle path option: keep literal black flags (onsite_required, budget_too_low) and
drop judgment-heavy ones (scope_misalignment, off_the_shelf_software).

## After Relabeling Completes

### 1. Evaluate Phase 1 + Phase 2 results

```bash
# Quick stats
python3 -c "
import json, collections
counts = collections.Counter()
total = flagged = 0
with open('data/train_relabel_2.jsonl') as f:
    for line in f:
        rec = json.loads(line.strip())
        total += 1
        if rec['flags']:
            flagged += 1
            for flag in rec['flags']:
                counts[flag] += 1
print(f'Total: {total} | Flagged: {flagged} | No-flag: {total - flagged}')
print('Flag distribution:')
for flag, count in counts.most_common():
    print(f'  {flag}: {count}')
"
```

Watch for:
- **Flag distribution** — any flag with 0 or very few hits may not be viable
- **Phase 2 fix rate** — >30% suggests Pass 1 too loose; <5% suggests Pass 2 not
  adding value
- **scope_misalignment FP rate** — spot-check a random sample of flagged chunks
- **Duplicate chunks** — some RFPs contain the same content twice (PDF artifacts).
  `build_training_set.py` deduplicates by chunk_text.

### 2. Update system prompt flag list

`scripts/system-prompt.md` has the old 14-flag list. Needs updating to match the
current 18 flags before retraining.

### 3. Build training set and retrain

```bash
PYTHONPATH=. uv run python scripts/build_training_set.py --input data/train_relabel_2.jsonl --no-rag --negative-ratio 0.2 --split 90/10/0
PYTHONPATH=. uv run python scripts/train.py --epochs 9 --max-seq-length 2048 --batch-size 8 --gradient-accumulation-steps 2
```

### 4. Relabel test set

`data/test_reserve.jsonl` uses the old 14-flag prompt and is obsolete. Options:
- Update `scripts/label_test_set.py` to match the new 18-flag prompt
- Hold out a subset by rfp_id from `train_relabel_2.jsonl` during the build step

## File Reference

| File | Role |
|------|------|
| `scripts/relabel_training_data.py` | The relabeling script (both phases) |
| `.development-notes/notes/test-labeling-prompt.md` | Source of truth for prompt text |
| `data/relabeled_pass1.jsonl` | Phase 1 output (intermediary) |
| `data/train_relabel_2.jsonl` | Phase 2 output (final, input to build_training_set.py) |
| `scripts/build_training_set.py` | Builds train/eval splits from Phase 2 output |
| `scripts/system-prompt.md` | System prompt for training (needs flag list update) |
| `data/relabeled_pass1a.jsonl` | Old Phase 1 run, kept for comparison (can delete) |

Source RFPs: `~/library 2/downloads/` (958 directories).
