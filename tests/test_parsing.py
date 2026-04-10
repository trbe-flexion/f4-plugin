from src.domain.parsing import parse_flags


class TestParseFlags:
    def test_single_flag(self):
        assert parse_flags("waterfall_methodology") == ["waterfall_methodology"]

    def test_multiple_flags(self):
        raw = "off_the_shelf_software\nno_custom_development"
        assert parse_flags(raw) == ["off_the_shelf_software", "no_custom_development"]

    def test_no_flag_returns_empty(self):
        assert parse_flags("no_flag") == []

    def test_empty_string(self):
        assert parse_flags("") == []

    def test_whitespace_only(self):
        assert parse_flags("   \n  ") == []

    def test_strips_whitespace(self):
        assert parse_flags("  brownfield  \n  lpta_source_selection  ") == [
            "brownfield",
            "lpta_source_selection",
        ]

    def test_filters_unknown_flags(self):
        raw = "waterfall_methodology\nmade_up_flag\nagile_methodology"
        result = parse_flags(raw)
        assert result == ["waterfall_methodology", "agile_methodology"]
        assert "made_up_flag" not in result

    def test_mixed_valid_and_no_flag(self):
        raw = "brownfield\nno_flag\nagile_methodology"
        result = parse_flags(raw)
        assert result == ["brownfield", "agile_methodology"]

    def test_garbage_output(self):
        assert parse_flags("I found several concerning issues with this RFP") == []

    def test_blank_lines_ignored(self):
        raw = "brownfield\n\n\nagile_methodology\n"
        result = parse_flags(raw)
        assert result == ["brownfield", "agile_methodology"]
