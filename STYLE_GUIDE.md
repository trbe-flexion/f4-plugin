# Writing Style Guide

This guide applies to ADRs (Architectural Decision Records/Action Plans) and reflection documents. Follow minimal styling conventions for easy grading.

Structure: For ADR section structure, see adr_template.md.

Length: Keep files under 50 lines. Up to 75 lines is acceptable if required, but aim for brevity.

Voice: Write in first person (I/my/me) as these document personal planning and decision-making. This applies to both ADRs and reflections.

Formatting: Use minimal styling. Avoid bold text for labels, section headers, or list items.

Good: "GPU: NVIDIA L4"
Bad: "**GPU**: NVIDIA L4"

Good: "Model: Llama 3.2 1B"
Bad: "**Model**: Llama 3.2 1B"

Paragraphs: Keep paragraphs short (2-4 sentences). This makes documents easier to read and grade.

Bullet Points: Use sparingly. Only use bullet points when they are overwhelmingly better than prose. Most content should be short paragraphs with clear topic sentences.

Headings: Use markdown headings (#, ##, ###) for section structure, but do not bold regular text or list labels.

Example:
```markdown
## Training Configuration

Model: meta-llama/Llama-3.2-1B
Dataset: Salesforce/wikitext
Batch size: 8
Learning rate: 2e-5

I chose these values because they balance training speed with model quality. The 2e-5 learning rate prevents overfitting on the small dataset.
```
