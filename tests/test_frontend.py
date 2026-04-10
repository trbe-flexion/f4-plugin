"""Tests for Gradio frontend app."""

from unittest.mock import MagicMock, patch

import gradio as gr

from src.domain.entities import FilterResult
from src.frontend.app import (
    _capture_logs,
    _format_details,
    _format_result,
    _make_handler,
    build_app,
)


class TestFormatResult:
    def test_filter_true(self):
        result = _format_result(True)
        assert "FILTER" in result

    def test_filter_false(self):
        result = _format_result(False)
        assert "PASS" in result


class TestFormatDetails:
    def test_includes_file_count(self):
        details = _format_details("some log", 1000, 3)
        assert "Files processed: 3" in details

    def test_includes_text_length(self):
        details = _format_details("some log", 12345, 1)
        assert "12,345" in details

    def test_includes_log_output(self):
        details = _format_details("F4 pipeline complete: 5 chunks", 100, 1)
        assert "F4 pipeline complete: 5 chunks" in details

    def test_empty_log(self):
        details = _format_details("", 100, 1)
        assert "no log output" in details


class TestCaptureLogsOutput:
    def test_captures_pipeline_logs(self):
        mock_pipeline = MagicMock()
        mock_pipeline.filter.return_value = FilterResult(filter=False)

        should_filter, log_output = _capture_logs(mock_pipeline, "some text")

        assert should_filter is False
        mock_pipeline.filter.assert_called_once_with("some text")


class TestMakeHandler:
    def test_no_files(self):
        mock_pipeline = MagicMock()
        handler = _make_handler(mock_pipeline)
        result, details = handler(None)
        assert "No files" in result

    def test_empty_files(self):
        mock_pipeline = MagicMock()
        handler = _make_handler(mock_pipeline)
        result, details = handler([])
        assert "No files" in result

    @patch("src.frontend.app.extract_and_combine")
    def test_successful_analysis(self, mock_extract):
        mock_extract.return_value = "RFP text content"
        mock_pipeline = MagicMock()
        mock_pipeline.filter.return_value = FilterResult(filter=True)

        handler = _make_handler(mock_pipeline)
        mock_file = MagicMock()
        mock_file.name = "/tmp/test.pdf"
        result, details = handler([mock_file])

        assert "FILTER" in result
        mock_pipeline.filter.assert_called_once_with("RFP text content")

    @patch("src.frontend.app.extract_and_combine")
    def test_extraction_error(self, mock_extract):
        mock_extract.side_effect = ValueError("Unsupported file type: .xyz")
        mock_pipeline = MagicMock()

        handler = _make_handler(mock_pipeline)
        mock_file = MagicMock()
        mock_file.name = "/tmp/test.xyz"
        result, details = handler([mock_file])

        assert "Extraction error" in result

    @patch("src.frontend.app.extract_and_combine")
    def test_empty_extraction(self, mock_extract):
        mock_extract.return_value = "   "
        mock_pipeline = MagicMock()

        handler = _make_handler(mock_pipeline)
        mock_file = MagicMock()
        mock_file.name = "/tmp/test.pdf"
        result, details = handler([mock_file])

        assert "No text could be extracted" in result


class TestBuildApp:
    def test_returns_blocks(self):
        mock_pipeline = MagicMock()
        app = build_app(mock_pipeline)
        assert isinstance(app, gr.Blocks)
