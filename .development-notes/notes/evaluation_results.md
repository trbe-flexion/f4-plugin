# F4 Evaluation Results

## Run 5: Data cleaning pass (4/15)

Model: meta-llama/Llama-3.2-3B-Instruct
Fine-tuning: LoRA, 5 epochs, record-level stratified split, max_seq_length 2048
Test set: 132 real RFP chunks
Training config: No RAG context, negatives at 0.25 ratio, brownfield dropped,
budget_too_low + large_team cleaned and oversampled, OTSS implicit cull from Run 4
Environment: SageMaker ml.g6.xlarge (L4 24GB)

| Metric | Value |
|--------|-------|
| Chunks | 132 |
| Format compliance | 100.0% |
| Precision | 80.3% |
| Recall | 53.0% |
| F1 | 63.9% |
| Predicted flags | 66 |
| Ground truth flags | 100 |
| Correct | 53 |

| Flag | Prec | Rec | TP | FP | FN |
|------|------|-----|----|----|-----|
| 8a_set_aside | 100.0% | 60.0% | 3 | 0 | 2 |
| agile_methodology | 100.0% | 36.4% | 4 | 0 | 7 |
| budget_too_low | 100.0% | 25.0% | 1 | 0 | 3 |
| design_exercise | 0.0% | 0.0% | 0 | 0 | 6 |
| hubzone_set_aside | 100.0% | 100.0% | 3 | 0 | 0 |
| large_team | 0.0% | 0.0% | 0 | 0 | 4 |
| lpta_source_selection | 100.0% | 75.0% | 6 | 0 | 2 |
| marginal_short_duration | 0.0% | 0.0% | 0 | 0 | 10 |
| off_the_shelf_software | 66.7% | 33.3% | 2 | 1 | 4 |
| onsite_required | 52.9% | 90.0% | 9 | 8 | 1 |
| oral_presentation | 100.0% | 84.6% | 11 | 0 | 2 |
| sdvosb_set_aside | 100.0% | 50.0% | 2 | 0 | 2 |
| small_business_set_aside | 100.0% | 66.7% | 8 | 0 | 4 |
| wosb_set_aside | 50.0% | 100.0% | 4 | 4 | 0 |

**Data per flag (at time of training):**

| Flag | Tier | Total | Real | Synth | Prec | Rec |
|------|------|------:|-----:|------:|-----:|----:|
| onsite_required | Black | 95 | 95 | 0 | 53% | 90% |
| off_the_shelf_software | Black | 53 | 53 | 0 | 67% | 33% |
| budget_too_low | Black | 39 | 19 | 20 | 100% | 25% |
| small_business_set_aside | Red | 137 | 137 | 0 | 100% | 67% |
| marginal_short_duration | Red | 100 | 100 | 0 | 0% | 0% |
| lpta_source_selection | Red | 98 | 98 | 0 | 100% | 75% |
| 8a_set_aside | Blue | 48 | 24 | 24 | 100% | 60% |
| large_team | Blue | 38 | 18 | 20 | 0% | 0% |
| sdvosb_set_aside | Blue | 37 | 37 | 0 | 100% | 50% |
| wosb_set_aside | Blue | 34 | 17 | 17 | 50% | 100% |
| hubzone_set_aside | Blue | 30 | 15 | 15 | 100% | 100% |
| oral_presentation | Green | 130 | 130 | 0 | 100% | 85% |
| design_exercise | Green | 54 | 54 | 0 | 0% | 0% |
| agile_methodology | Green | 123 | 123 | 0 | 100% | 36% |

**Changes from Run 4:** Dropped brownfield entirely (no chunk-level signal). Cleaned
budget_too_low (removed synthetic + no-dollar-amount examples, oversampled 20 from clean
seeds). Cleaned large_team (removed examples without explicit headcount/roles, oversampled
20 from clean seeds).

**Results vs Run 4:** Precision improved 60.8%→80.3% (+19.5). F1 improved 57.9%→63.9%
(+6.0). OTSS FP dropped 27→1. budget_too_low now firing (1 TP). However, recall dipped
55.4%→53.0%. New regressions: agile_methodology (73%→36%), marginal_short_duration
(36%→0%), onsite_required FP up (3→8). Model is more conservative overall (66 predicted
vs 102). large_team and design_exercise still at zero recall.

---

## Run 4: Record-level split, 5 epochs, OTSS cap 75 (4/14)

Model: meta-llama/Llama-3.2-3B-Instruct
Fine-tuning: LoRA, 5 epochs, record-level stratified split, max_seq_length 2048
Test set: 130 real RFP chunks
Training config: No RAG context, OTSS capped at 75, negatives at 0.25 ratio
System prompt: "weak signals" caveat removed
Environment: SageMaker ml.g6.xlarge (L4 24GB)

| Metric | Value |
|--------|-------|
| Chunks | 130 |
| Format compliance | 99.2% |
| Precision | 60.8% |
| Recall | 55.4% |
| F1 | 57.9% |
| Predicted flags | 102 |
| Ground truth flags | 112 |
| Correct | 62 |

| Flag | Prec | Rec | TP | FP | FN |
|------|------|-----|----|----|-----|
| 8a_set_aside | 100.0% | 75.0% | 3 | 0 | 1 |
| agile_methodology | 91.7% | 73.3% | 11 | 1 | 4 |
| brownfield | 0.0% | 0.0% | 0 | 0 | 10 |
| budget_too_low | 0.0% | 0.0% | 0 | 1 | 4 |
| design_exercise | 0.0% | 0.0% | 0 | 1 | 4 |
| hubzone_set_aside | 66.7% | 100.0% | 2 | 1 | 0 |
| large_team | 0.0% | 0.0% | 0 | 0 | 3 |
| lpta_source_selection | 60.0% | 50.0% | 3 | 2 | 3 |
| marginal_short_duration | 80.0% | 36.4% | 4 | 1 | 7 |
| off_the_shelf_software | 22.9% | 50.0% | 8 | 27 | 8 |
| onsite_required | 70.0% | 70.0% | 7 | 3 | 3 |
| oral_presentation | 87.5% | 87.5% | 7 | 1 | 1 |
| sdvosb_set_aside | 80.0% | 100.0% | 4 | 1 | 0 |
| small_business_set_aside | 91.7% | 84.6% | 11 | 1 | 2 |
| wosb_set_aside | 100.0% | 100.0% | 2 | 0 | 0 |

**Changes from Run 3:** OTSS cap reduced 150→75, record-level stratified split (was rfp_id),
5 epochs (was 3).

**Results vs Run 3:** F1 improved 48.6%→57.9% (+9.3). Precision 52.0%→60.8%, recall
45.5%→55.4%. Broad improvements across most flags. OTSS FP down slightly (29→27) but
still the main precision drag. Four flags remain at zero recall: brownfield, budget_too_low,
design_exercise, large_team — extra epochs and data changes had no effect on these.

---

## Run 3: Real data, no RAG (4/14)

There were additional runs between 2 and 3 using a RAG.
Compliance remains high through all runs (98%+)
Precision was high when using a RAG.
Recall, however, was low (25%). I reduced negative examples (no_flag) and removed prompt language instructing the model to ignore weak signals.
This run is the result of that adjustment: Much weaker precision driven significantly by OTSS false positives, moderately better Recall, and slightly better F1.
Result of this run: reduce OTSS examples; switch from splitting by RFP to a record-level stratified split. This may be worse for leakage, but better for
providing a wider selection of training examples for fine-tuning. I also decided to increase epochs to 5, as without RAG guidance the signal may be weaker.

Another idea not yet pursued: go the other way entirely. Change the training set into a large RAG and briefly train the model for output compliance.

Model: meta-llama/Llama-3.2-3B-Instruct
Fine-tuning: LoRA, 3 epochs, 1034 training examples, max_seq_length 2048
Test set: 130 real RFP chunks (stratified by rfp_id)
Training config: No RAG context, OTSS capped at 150, negatives at 0.25 ratio
System prompt: "weak signals" caveat removed
Environment: SageMaker ml.g6.xlarge (L4 24GB)

| Metric | Value |
|--------|-------|
| Chunks | 130 |
| Format compliance | 99.2% |
| Precision | 52.0% |
| Recall | 45.5% |
| F1 | 48.6% |
| Predicted flags | 98 |
| Ground truth flags | 112 |
| Correct | 51 |

| Flag | Prec | Rec | TP | FP | FN |
|------|------|-----|----|----|-----|
| 8a_set_aside | 100.0% | 75.0% | 3 | 0 | 1 |
| agile_methodology | 90.9% | 66.7% | 10 | 1 | 5 |
| brownfield | 0.0% | 0.0% | 0 | 0 | 10 |
| budget_too_low | 0.0% | 0.0% | 0 | 1 | 4 |
| design_exercise | 0.0% | 0.0% | 0 | 0 | 4 |
| hubzone_set_aside | 66.7% | 100.0% | 2 | 1 | 0 |
| large_team | 0.0% | 0.0% | 0 | 0 | 3 |
| lpta_source_selection | 40.0% | 33.3% | 2 | 3 | 4 |
| marginal_short_duration | 50.0% | 18.2% | 2 | 2 | 9 |
| off_the_shelf_software | 17.1% | 37.5% | 6 | 29 | 10 |
| onsite_required | 66.7% | 60.0% | 6 | 3 | 4 |
| oral_presentation | 87.5% | 87.5% | 7 | 1 | 1 |
| sdvosb_set_aside | 57.1% | 100.0% | 4 | 3 | 0 |
| small_business_set_aside | 87.5% | 53.8% | 7 | 1 | 6 |
| wosb_set_aside | 50.0% | 100.0% | 2 | 2 | 0 |

**Notes:** Recall improved dramatically vs Run 2 (25.5% → 45.5%) after cutting negatives
to 0.25 ratio. OTSS over-predicts (29 FP). Four flags still at zero recall (brownfield,
budget_too_low, design_exercise, large_team). Next: cut OTSS to 75, try 5 epochs.

---

## Run 2: Real data, with RAG (4/14)

Model: meta-llama/Llama-3.2-3B-Instruct
Fine-tuning: LoRA, 3 epochs, 1536 training examples, max_seq_length 2048
Test set: 191 real RFP chunks (stratified by rfp_id)
Training config: 30 real RAG seeds, OTSS capped at 100, negatives at 1:1 ratio
Environment: SageMaker ml.g6.xlarge (L4 24GB)

| Metric | Value |
|--------|-------|
| Chunks | 191 |
| Format compliance | 99.5% |
| Precision | 86.2% |
| Recall | 25.5% |
| F1 | 39.4% |
| Predicted flags | 29 |
| Ground truth flags | 98 |
| Correct | 25 |

| Flag | Prec | Rec | TP | FP | FN |
|------|------|-----|----|----|-----|
| 8a_set_aside | 100.0% | 33.3% | 1 | 0 | 2 |
| agile_methodology | 100.0% | 25.0% | 3 | 0 | 9 |
| brownfield | 0.0% | 0.0% | 0 | 0 | 4 |
| budget_too_low | 0.0% | 0.0% | 0 | 0 | 5 |
| design_exercise | 0.0% | 0.0% | 0 | 0 | 5 |
| hubzone_set_aside | 100.0% | 100.0% | 2 | 0 | 0 |
| large_team | 0.0% | 0.0% | 0 | 0 | 4 |
| lpta_source_selection | 100.0% | 16.7% | 1 | 0 | 5 |
| marginal_short_duration | 0.0% | 0.0% | 0 | 0 | 8 |
| off_the_shelf_software | 0.0% | 0.0% | 0 | 0 | 9 |
| onsite_required | 0.0% | 0.0% | 0 | 0 | 6 |
| oral_presentation | 81.8% | 90.0% | 9 | 2 | 1 |
| sdvosb_set_aside | 100.0% | 66.7% | 2 | 0 | 1 |
| small_business_set_aside | 87.5% | 38.9% | 7 | 1 | 11 |
| wosb_set_aside | 0.0% | 0.0% | 0 | 1 | 3 |

**Notes:** High precision but very low recall — model defaulted to no_flag too often.
Negatives at 1:1 ratio overwhelmed flag signal. RAG context (30 seeds) likely too sparse
to help.

---

## Run 1: Synthetic data (4/10)

Model: meta-llama/Llama-3.2-3B-Instruct
Fine-tuning: LoRA, 3 epochs, 1670 training examples, max_seq_length 2048
Test set: 202 held-out synthetic examples (~8 per flag)
Environment: SageMaker ml.g6.xlarge (L4 24GB)

## Summary

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Chunks | 202 | 202 |
| Format compliance | 85.2% | 100.0% |
| Precision | 5.8% | 97.3% |
| Recall | 32.9% | 91.1% |
| F1 | 9.9% | 94.1% |
| Predicted flags | 893 | 148 |
| Ground truth flags | 158 | 158 |
| Correct | 52 | 144 |

## Comparison

| Metric | Base | Fine-tuned | Delta |
|--------|------|------------|-------|
| Format compliance | 85.2% | 100.0% | +14.8% |
| Precision | 5.8% | 97.3% | +91.5% |
| Recall | 32.9% | 91.1% | +58.2% |
| F1 | 9.9% | 94.1% | +84.2% |

## Base Model — Per-Flag Results

| Flag | Prec | Rec | TP | FP | FN |
|------|------|-----|----|----|-----|
| 8a_set_aside | 9.8% | 75.0% | 6 | 55 | 2 |
| agile_methodology | 0.0% | 0.0% | 0 | 19 | 8 |
| brownfield | 50.0% | 12.5% | 1 | 1 | 7 |
| budget_too_low | 0.0% | 0.0% | 0 | 2 | 8 |
| design_exercise | 0.0% | 0.0% | 0 | 0 | 8 |
| edwosb_set_aside | 0.0% | 0.0% | 0 | 34 | 8 |
| hubzone_set_aside | 0.0% | 0.0% | 0 | 35 | 8 |
| large_team | 28.6% | 44.4% | 4 | 10 | 5 |
| lpta_source_selection | 0.0% | 0.0% | 0 | 109 | 8 |
| marginal_short_duration | 0.0% | 0.0% | 0 | 1 | 8 |
| no_custom_development | 5.3% | 100.0% | 9 | 160 | 0 |
| off_the_shelf_software | 5.5% | 100.0% | 8 | 137 | 0 |
| onsite_madison | 28.6% | 25.0% | 2 | 5 | 6 |
| onsite_required | 20.0% | 22.2% | 2 | 8 | 7 |
| oral_presentation | 83.3% | 50.0% | 5 | 1 | 5 |
| sdvosb_set_aside | 2.9% | 12.5% | 1 | 33 | 7 |
| small_business_set_aside | 12.0% | 66.7% | 6 | 44 | 3 |
| waterfall_methodology | 4.9% | 100.0% | 8 | 154 | 0 |
| wosb_set_aside | 0.0% | 0.0% | 0 | 33 | 8 |

## Fine-Tuned Model — Per-Flag Results

| Flag | Prec | Rec | TP | FP | FN |
|------|------|-----|----|----|-----|
| 8a_set_aside | 100.0% | 100.0% | 8 | 0 | 0 |
| agile_methodology | 100.0% | 100.0% | 8 | 0 | 0 |
| brownfield | 100.0% | 75.0% | 6 | 0 | 2 |
| budget_too_low | 100.0% | 100.0% | 8 | 0 | 0 |
| design_exercise | 100.0% | 100.0% | 8 | 0 | 0 |
| edwosb_set_aside | 100.0% | 100.0% | 8 | 0 | 0 |
| hubzone_set_aside | 100.0% | 100.0% | 8 | 0 | 0 |
| large_team | 100.0% | 77.8% | 7 | 0 | 2 |
| lpta_source_selection | 100.0% | 100.0% | 8 | 0 | 0 |
| marginal_short_duration | 100.0% | 87.5% | 7 | 0 | 1 |
| no_custom_development | 100.0% | 88.9% | 8 | 0 | 1 |
| off_the_shelf_software | 100.0% | 75.0% | 6 | 0 | 2 |
| onsite_madison | 100.0% | 75.0% | 6 | 0 | 2 |
| onsite_required | 69.2% | 100.0% | 9 | 4 | 0 |
| oral_presentation | 100.0% | 80.0% | 8 | 0 | 2 |
| sdvosb_set_aside | 100.0% | 100.0% | 8 | 0 | 0 |
| small_business_set_aside | 100.0% | 88.9% | 8 | 0 | 1 |
| waterfall_methodology | 100.0% | 87.5% | 7 | 0 | 1 |
| wosb_set_aside | 100.0% | 100.0% | 8 | 0 | 0 |
