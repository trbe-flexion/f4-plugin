# Real Data Generation Notes

Notes from the first full labeling run (`scripts/label_real_data.py`, 958 RFPs, Sonnet via Bedrock).

---

## Quality Spot Check

Two random samples of 30 chunks each, read in full and evaluated.

**Error rate: ~13% across both samples (8 mislabels out of 60).**

### Error-Prone Flags

- `no_custom_development` — Most frequent mislabel. Claude applies it when an RFP is *purchasing*
  a product (CrowdStrike subscription, Pluralsight licenses) rather than when the RFP *explicitly
  excludes* custom development. Buying a product ≠ "no custom development." Should often be
  `off_the_shelf_software` or `no_flag`.
- `large_team` — Fires on user counts (15,000 users), vendor counts (5 vendors), CONUS site
  counts, and other numbers that aren't team size. The flag definition says "10 or more personnel"
  but Claude interprets any large number as large team.
- `marginal_short_duration` — Boundary precision issue. Labeled a 12-month base period as
  "less than 12 months" (12 is not less than 12). Also labeled a 16-month PoP incorrectly.
- `waterfall_methodology` — All 7 examples found were medium confidence and most were mislabels.
  Claude triggered on project phases, contract phases, and hardware installation sequences — none
  were actual waterfall SDLC mandates. See waterfall section below.
- `brownfield` — Sometimes applied to system *replacement* (e.g., replacing voting tabulators)
  rather than taking over an existing codebase. Replacement ≠ brownfield.

### What Works Well

- High-confidence labels are substantially better than medium-confidence
- Set-aside flags (small_business, sdvosb, 8a, etc.) are reliable — the language is formulaic
- Adversarial labels are generally correct — Claude identifies near-misses well
- `off_the_shelf_software` labels are accurate when applied
- `agile_methodology` labels are accurate — explicit keyword matches
- `no_flag` labels are almost always correct

### Mitigation Options

1. Drop medium-confidence chunks (free, immediate, loses volume)
2. Tighten flag definitions in the prompt for error-prone flags
3. Validation pass: second Claude call on extracted chunks to verify labels
4. Post-filter rules in `build_training_set.py`

---

## Waterfall Language in Real RFPs

Waterfall methodology was rare in real RFP labeling (4/150 at 30% through corpus).
This may be genuinely rare in Flexion's pipeline, or the detection language may be too narrow.

### Current Detection Phrases (from collated-flag-set.md)
- "waterfall"
- "traditional SDLC"
- "sequential phases"
- "standard software lifecycle development"
- "fixed requirements baseline"
- "phase gate"
- "V-model"

### Expanded Phrases to Look For
- "phased approach" / "phased delivery"
- "requirements must be finalized before development begins"
- "complete requirements document prior to design"
- "linear development process"
- "SDLC" (without "traditional" qualifier)
- "milestone-based delivery" with sequential gates
- "development will not commence until requirements are approved"
- "big bang deployment" / "single release"
- "MIL-STD-498" (DoD waterfall standard)
- "IEEE 12207" lifecycle process
- "spiral" (sometimes used interchangeably with waterfall in gov context)

### Implicit Waterfall (No Keyword)
RFPs that describe waterfall without naming it:
- "all requirements shall be documented and baselined before any coding begins"
- Sequential phase descriptions with gate reviews between each
- Requirements freeze language before design/development

---

## Rare Flag Counts (at ~30% / 287 RFPs)

| Flag | Count | Likely to fill? |
|------|-------|-----------------|
| onsite_madison | 1 | No |
| waterfall_methodology | 4 | No |
| edwosb_set_aside | 6 | No |
| budget_too_low | 8 | No |
| wosb_set_aside | 17 | Unlikely |
| 8a_set_aside | 18 | Unlikely |
| hubzone_set_aside | 18 | Unlikely |

Plan: supplement rare flags with seeded synthetic examples using the real examples
as style/language seeds for Claude generation.

---

## OTSS Implicit Example Cull (4/15)

Removed `off_the_shelf_software` from 212 records in `opus_validated_real.jsonl` that lacked
explicit COTS terminology. Kept 54 records with explicit language.

**Problem:** OTSS had 27 FP in Run 4 eval. ~81% of training examples used implicit language
(SaaS, cloud-based, configuration, vendor platform names) rather than explicit COTS terms.
The model learned to associate any commercial software mention with the flag, producing FPs
on chunks that mention software incidentally.

**Decision:** OTSS is a wide-context flag similar to `onsite_madison` — determining whether
a procurement is *for* off-the-shelf software often requires document-level understanding.
Keep narrow, explicit detections in the model; handle the rest via string-match post-filter
at inference time.

**Kept terms** (case-insensitive):
- COTS, GOTS, MCOTS (word-boundary matched)
- commercial off-the-shelf (any hyphenation)
- off-the-shelf software (any hyphenation)
- NDI / nondevelopmental item
- OOTB / out-of-the-box
- FAR Part 12
- shrink-wrapped

**Result:** 266 OTSS records → 54 explicit kept, 212 implicit removed (flag stripped;
records with other flags retained, records left with no flags set to `no_flag`).

---

## Zero-Recall Flag Investigation (4/15)

Four flags showed 0% recall across Runs 3 and 4: brownfield, budget_too_low, design_exercise, large_team.
Training counts were adequate (35-73 per flag) so this is a data quality problem, not volume.

### brownfield — DROP

- 100 non-dropped examples, but language is too diffuse for chunk-level detection
- No single keyword appears in even half the examples: "maintain" (35%), "transition" (31%),
  "moderniz" (24%) were the closest
- Actual brownfield-specific terms ("existing code", "codebase", "take over") appear in <3%
- 46% of examples were "fixed" by Opus — Sonnet's original labels needed substantial correction
- The flag covers a broad concept ("there's already something here") that requires document-level
  understanding, similar to onsite_madison
- Decision: drop from model, handle at document level

### budget_too_low — CLEAN + OVERSAMPLE

- 46 non-dropped examples, but 23 (50%) are synthetic with uniform format that doesn't match
  real RFP language. Drop all synthetic examples.
- Of 23 real examples, ~6 contain no dollar amount at all — budget is inferred from scope
  (training courses, bus cameras, 2-day seminars). These dilute the signal. Remove them.
- Remaining ~17 real examples have a consistent, learnable signal: explicit dollar amount
  + total/ceiling/NTE framing, under $100K.
- Oversample using Claude generation seeded from the clean real examples:
  - Rotate 2-3 seed examples per generation call (don't use all 17 at once)
  - Tell Claude the target signal explicitly: "chunk contains an explicit total contract
    value/ceiling/NTE under $100K"
  - Vary surface features (agencies, contract types, formatting) while keeping core signal

### design_exercise — NO ACTION, RE-EVALUATE

- 54 non-dropped examples, 94% high confidence, 81% "keep" — cleanest data of the four.
- Strong keyword signal: "demonstrat/demo" 76%, "challenge" 41%, "POC" 30%, "presentation" 50%.
- 41 in train, 40 with clear keywords. Eval (6) and test (5) all have keywords too.
- Data quality is not the problem. Similar-count flags (hubzone 34, wosb 42) work fine.
- Hypothesis: improved after dropping brownfield/cleaning budget_too_low/large_team changes
  the training distribution enough for this flag to start firing.
- Re-evaluate after next training run. If still zero, consider oversampling then.

### large_team — CLEAN + OVERSAMPLE

- 54 non-dropped examples, but only 18 (33%) have clear chunk-level signal: explicit
  headcount ≥10 (e.g. "13 FTEs", "104 contractor-personnel", "200 FTE") or 10+ enumerated
  role listings.
- Remaining 36 are noise: labor hours (488,994 hours), vague staffing plan language, team
  descriptions without counts, concurrent user counts mistaken for team size.
- Known labeling issue from gen notes: Claude fires on any large number, not just team size.
- 31% medium confidence, 31% fixed by Opus — consistent with noisy labeling.
- Clean to keep only examples with explicit personnel/FTE counts ≥10 or long role enumerations.
- Oversample using Claude generation seeded from the ~18 clean real examples:
  - Rotate 2-3 seeds per generation call
  - Tell Claude the target signal: "chunk explicitly states or enumerates 10+ contractor
    personnel/FTEs/roles required"
  - Vary agencies, contract types, role names while keeping core signal

---

## Other Observations

- No-flag count overshot target (460/150 at 30%) because the prompt always asks for
  1-2 no-flag passages per section. Not harmful — can downsample in build_training_set.py.
- JSON parse failures: ~3-5% of API calls, mostly on large RFPs. Script handles gracefully.
- Duplicate chunks from overlapping sections on split RFPs. Dedup added to build_training_set.py.
