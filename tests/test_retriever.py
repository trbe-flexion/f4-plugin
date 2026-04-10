"""Tests for RAG retriever context formatting."""

from src.rag.retriever import format_context


class TestFormatContext:
    def test_single_result(self):
        results = [{"passage": "This uses waterfall.", "flag": "waterfall_methodology"}]
        output = format_context(results, "Some RFP chunk text")
        assert "[Retrieved context]" in output
        assert "Example 1:" in output
        assert "This uses waterfall." in output
        assert "RFP chunk to analyze:" in output
        assert "Some RFP chunk text" in output

    def test_multiple_results(self):
        results = [
            {"passage": "First passage.", "flag": "brownfield"},
            {"passage": "Second passage.", "flag": "agile_methodology"},
            {"passage": "Third passage.", "flag": "waterfall_methodology"},
        ]
        output = format_context(results, "chunk")
        assert "Example 1:" in output
        assert "Example 2:" in output
        assert "Example 3:" in output
        assert "First passage." in output
        assert "Third passage." in output

    def test_empty_results(self):
        output = format_context([], "chunk text")
        assert "[Retrieved context]" in output
        assert "RFP chunk to analyze:" in output
        assert "chunk text" in output
        assert "Example" not in output

    def test_separator_present(self):
        results = [{"passage": "p", "flag": "f"}]
        output = format_context(results, "chunk")
        assert "---" in output

    def test_chunk_after_separator(self):
        results = [{"passage": "p", "flag": "f"}]
        output = format_context(results, "my chunk")
        parts = output.split("---")
        assert len(parts) == 2
        assert "my chunk" in parts[1]
