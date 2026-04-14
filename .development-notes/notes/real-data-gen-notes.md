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

## Other Observations

- No-flag count overshot target (460/150 at 30%) because the prompt always asks for
  1-2 no-flag passages per section. Not harmful — can downsample in build_training_set.py.
- JSON parse failures: ~3-5% of API calls, mostly on large RFPs. Script handles gracefully.
- Duplicate chunks from overlapping sections on split RFPs. Dedup added to build_training_set.py.
