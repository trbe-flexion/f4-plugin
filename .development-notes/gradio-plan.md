# Gradio Frontend Plan

## Context

Checklist step 12. Demo surface for F4. Uploads PDF/DOCX, extracts text, concatenates multi-file packages (matching opp-capture's pattern), calls pipeline.filter(), displays filter decision plus pretty-printed pipeline logs. Password-protected, share=True tunnel for demo.

## Decisions

- Gradio Blocks, not ChatInterface
- File upload: PDF + DOCX, multiple files
- Text extraction: pdfplumber (PDF), python-docx (DOCX)
- Multi-file concat: labeled headers matching opp-capture format
- Log capture: temporary handler on pipeline logger, displayed in details panel
- Auth + share=True for demo
- Pipeline injected into app (constructor injection)

## Files Created

- src/frontend/app.py — Gradio Blocks app
- src/frontend/extraction.py — text extraction + multi-file concat
- tests/test_frontend.py, tests/test_extraction.py

## Files Modified

- pyproject.toml — gradio, pdfplumber, python-docx deps

## Steps

1. Save plan
2. Add deps
3. Text extraction (extract_text, extract_and_combine)
4. Gradio app (build_app, launch_app, log capture)
5. Tests (80%+ coverage, mock pipeline)
6. Lint, verify, update checklist
