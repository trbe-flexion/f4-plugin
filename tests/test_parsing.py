from src.domain.parsing import parse_flags


class TestParseFlags:
    def test_single_flag(self):
        assert parse_flags("lpta_source_selection") == ["lpta_source_selection"]

    def test_multiple_flags(self):
        raw = "agile_methodology\noral_presentation"
        assert parse_flags(raw) == ["agile_methodology", "oral_presentation"]

    def test_no_flag_returns_empty(self):
        assert parse_flags("no_flag") == []

    def test_empty_string(self):
        assert parse_flags("") == []

    def test_whitespace_only(self):
        assert parse_flags("   \n  ") == []

    def test_strips_whitespace(self):
        assert parse_flags("  small_business_set_aside  \n  lpta_source_selection  ") == [
            "small_business_set_aside",
            "lpta_source_selection",
        ]

    def test_filters_unknown_flags(self):
        raw = "lpta_source_selection\nmade_up_flag\nagile_methodology"
        result = parse_flags(raw)
        assert result == ["lpta_source_selection", "agile_methodology"]
        assert "made_up_flag" not in result

    def test_mixed_valid_and_no_flag(self):
        raw = "small_business_set_aside\nno_flag\nagile_methodology"
        result = parse_flags(raw)
        assert result == ["small_business_set_aside", "agile_methodology"]

    def test_garbage_output(self):
        assert parse_flags("I found several concerning issues with this RFP") == []

    def test_blank_lines_ignored(self):
        raw = "small_business_set_aside\n\n\nagile_methodology\n"
        result = parse_flags(raw)
        assert result == ["small_business_set_aside", "agile_methodology"]

    def test_none_returns_empty(self):
        assert parse_flags("none") == []

    def test_none_capitalized_returns_empty(self):
        assert parse_flags("None") == []

    def test_comma_separated(self):
        raw = "lpta_source_selection, small_business_set_aside"
        result = parse_flags(raw)
        assert result == ["lpta_source_selection", "small_business_set_aside"]

    def test_comma_and_newline_mixed(self):
        raw = "lpta_source_selection, small_business_set_aside\nagile_methodology"
        result = parse_flags(raw)
        assert result == ["lpta_source_selection", "small_business_set_aside", "agile_methodology"]

    def test_comma_separated_with_prose(self):
        raw = "lpta_source_selection, some junk, agile_methodology"
        result = parse_flags(raw)
        assert result == ["lpta_source_selection", "agile_methodology"]

    def test_none_no_flag_returns_empty(self):
        assert parse_flags("None, no_flag") == []

    def test_no_flag_case_insensitive(self):
        assert parse_flags("NO_FLAG") == []
