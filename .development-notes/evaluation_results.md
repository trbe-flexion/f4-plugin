# F4 Evaluation Results

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
