from src.decision.engine import FilterDecisionEngine


class TestFilterDecisionEngine:
    def test_no_flags_no_filter(self):
        engine = FilterDecisionEngine()
        assert engine.decide(set()) is False

    def test_red_flag_below_threshold(self):
        engine = FilterDecisionEngine(red_flag_threshold=3)
        assert engine.decide({"lpta_source_selection"}) is False

    def test_red_flags_at_threshold(self):
        engine = FilterDecisionEngine(red_flag_threshold=2)
        assert engine.decide({"lpta_source_selection", "small_business_set_aside"}) is True

    def test_red_flags_above_threshold(self):
        engine = FilterDecisionEngine(red_flag_threshold=1)
        assert engine.decide({"lpta_source_selection", "small_business_set_aside"}) is True

    def test_default_threshold_never_triggers_red(self):
        engine = FilterDecisionEngine()
        all_red = {"lpta_source_selection", "small_business_set_aside"}
        assert engine.decide(all_red) is False

    def test_green_flags_no_filter(self):
        engine = FilterDecisionEngine()
        assert engine.decide({"agile_methodology", "oral_presentation"}) is False

    def test_blue_flags_no_filter(self):
        engine = FilterDecisionEngine()
        assert engine.decide({"8a_set_aside", "hubzone_set_aside"}) is False

    def test_unknown_flags_ignored(self):
        engine = FilterDecisionEngine()
        assert engine.decide({"totally_fake_flag"}) is False
