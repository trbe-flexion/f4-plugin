"""Gradio frontend for F4 flag detection demo."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import gradio as gr

from src.frontend.extraction import extract_and_combine

if TYPE_CHECKING:
    from src.pipeline.filter import F4Pipeline

PIPELINE_LOGGER_NAME = "src.pipeline.filter"


def _capture_logs(pipeline: F4Pipeline, text: str) -> tuple[bool, str]:
    """Run pipeline.filter() while capturing log output.

    Returns (filter_decision, log_output).
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(message)s"))

    pipeline_logger = logging.getLogger(PIPELINE_LOGGER_NAME)
    original_level = pipeline_logger.level
    pipeline_logger.addHandler(handler)
    pipeline_logger.setLevel(logging.INFO)

    try:
        result = pipeline.filter(text)
    finally:
        pipeline_logger.removeHandler(handler)
        pipeline_logger.setLevel(original_level)

    return result.filter, buffer.getvalue()


def _format_result(should_filter: bool) -> str:
    """Format the filter decision for display."""
    if should_filter:
        return "FILTER — This RFP does not appear to be a good fit."
    return "PASS — No disqualifying flags detected. Proceed to full analysis."


def _format_details(log_output: str, text_length: int, file_count: int) -> str:
    """Format pipeline details for the details panel."""
    lines = [
        f"Files processed: {file_count}",
        f"Total text length: {text_length:,} characters",
        "",
        "Pipeline log:",
        log_output if log_output.strip() else "  (no log output)",
    ]
    return "\n".join(lines)


def _make_handler(pipeline: F4Pipeline):
    """Create the analyze handler function."""

    def analyze(files):
        if not files:
            return "No files uploaded.", ""

        file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]

        try:
            text = extract_and_combine(file_paths)
        except ValueError as e:
            return f"Extraction error: {e}", ""
        except Exception as e:
            return f"Error reading files: {e}", ""

        if not text.strip():
            return "No text could be extracted from the uploaded files.", ""

        should_filter, log_output = _capture_logs(pipeline, text)

        result_text = _format_result(should_filter)
        details = _format_details(log_output, len(text), len(file_paths))

        return result_text, details

    return analyze


def build_app(pipeline: F4Pipeline) -> gr.Blocks:
    """Build the Gradio Blocks app."""
    with gr.Blocks(title="F4 — RFP Flag Detection") as app:
        gr.Markdown("# F4 — Flexion Fast Fail Filtering")
        gr.Markdown(
            "Upload RFP documents (PDF/DOCX) to screen for disqualifying flags "
            "before expensive LLM analysis."
        )

        file_input = gr.File(
            label="Upload RFP documents",
            file_count="multiple",
            file_types=[".pdf", ".docx"],
        )

        with gr.Row():
            analyze_btn = gr.Button("Analyze", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

        result_output = gr.Textbox(
            label="Decision",
            interactive=False,
            lines=2,
        )

        details_output = gr.Textbox(
            label="Details",
            interactive=False,
            lines=12,
        )

        handler = _make_handler(pipeline)
        analyze_btn.click(
            fn=handler,
            inputs=[file_input],
            outputs=[result_output, details_output],
        )
        clear_btn.click(
            fn=lambda: (None, "", ""),
            inputs=[],
            outputs=[file_input, result_output, details_output],
        )

    return app


def launch_app(
    pipeline: F4Pipeline,
    auth: tuple[str, str] | None = None,
    share: bool = False,
) -> None:
    """Build and launch the Gradio app."""
    app = build_app(pipeline)
    launch_kwargs: dict = {"share": share}
    if auth:
        launch_kwargs["auth"] = [auth]
    app.launch(**launch_kwargs)
