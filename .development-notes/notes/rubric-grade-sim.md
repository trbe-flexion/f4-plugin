# Rubric Self-Assessment (Conservative)

Assessed: 2026-04-16, post-README update

## I. Model & Inference (40 pts)

### Model Functionality (20 pts) — 17-18

LoRA fine-tuned Llama 3.2 3B running on Bedrock, 88.7% F1 on held-out test. Thorough evaluation across 8+ cycles with per-flag breakdowns. Docking points because: test set is derived from the same corpus as training (Opus-labeled real data, not truly independent ground truth), and the final model covers 7 of the original 30+ flags.

### Innovation & Creativity (20 pts) — 16-17

Real production context, navigated real deployment constraints (Bedrock CMI, no constrained generation, RoPE debugging). Specialist-model and pipeline-integrated-data-gathering ideas show vision. But: model isn't in production yet, RAG was built and abandoned, TextGrad was planned and not attempted. The stretch is real but some of it lives in the retrospective rather than in working code.

## II. Production Environment & Programmability (30 pts)

### Environment Setup (15 pts) — 13-14

Model deployed to Bedrock, endpoint accessible, Gradio frontend working end-to-end. No Terraform (manual deployment), Alt account only (not migrated to Main). But it works and is demonstrable.

### Inference Pipeline (15 pts) — 13-14

Full pipeline: chunking -> concurrency -> parse/retry -> deduplication -> decision logic. 89% test coverage. Rubric mentions "appropriate sampling method" — inference uses temperature/top_p via Bedrock but there's no explicit discussion of sampling strategy in documentation. Minor gap depending on how professor weighs it.

## III. Documentation & Presentation (30 pts)

### Technical Documentation (15 pts) — 11-13

Updated README covers model selection, training config, eval results, pipeline flow, deployment, and project structure. Supporting docs: ADR, evaluation_results.md (8 runs), bedrock-deployment.md, collated-flag-set.md, retrospective.md.

Gaps:
- No programmatic usage snippet showing how opp-capture would call F4Pipeline.filter()
- Sampling parameters (temperature, top_p) not documented anywhere
- Documentation is spread across many files rather than a single consolidated write-up

### Demo & Presentation (15 pts) — 12-14

Script is solid: real context, clear progression, live demo hitting real endpoint, honest retrospective with forward-looking strategy. Range depends on delivery.

## Summary

| Section | Points | Rating |
|---------|--------|--------|
| Model Functionality (20) | 17-18 | Excellent |
| Innovation & Creativity (20) | 16-17 | Excellent |
| Environment Setup (15) | 13-14 | Excellent |
| Inference Pipeline (15) | 13-14 | Excellent |
| Technical Docs (15) | 11-13 | Good-Excellent |
| Demo & Presentation (15) | 12-14 | Excellent |
| **Total** | **82-90** | |

## Quick Wins to Close Gaps

1. Add a programmatic usage snippet to README showing F4Pipeline.filter() call
2. Document sampling parameters (temperature, top_p) and rationale — even one sentence
3. Both address the "API" and "sampling method" rubric language directly
