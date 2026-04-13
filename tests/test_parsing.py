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

    def test_none_returns_empty(self):
        assert parse_flags("none") == []

    def test_none_capitalized_returns_empty(self):
        assert parse_flags("None") == []

    def test_comma_separated(self):
        raw = "waterfall_methodology, onsite_required"
        result = parse_flags(raw)
        assert result == ["waterfall_methodology", "onsite_required"]

    def test_comma_and_newline_mixed(self):
        raw = "waterfall_methodology, onsite_required\nagile_methodology"
        result = parse_flags(raw)
        assert result == ["waterfall_methodology", "onsite_required", "agile_methodology"]

    def test_comma_separated_with_prose(self):
        raw = "waterfall_methodology, some junk, agile_methodology"
        result = parse_flags(raw)
        assert result == ["waterfall_methodology", "agile_methodology"]

    def test_none_no_flag_returns_empty(self):
        assert parse_flags("None, no_flag") == []

    def test_no_flag_case_insensitive(self):
        assert parse_flags("NO_FLAG") == []
