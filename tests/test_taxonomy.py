from src.domain.taxonomy import (
    BLUE_FLAGS,
    FLAG_TIERS,
    GREEN_FLAGS,
    RED_FLAGS,
    VALID_FLAGS,
)


class TestFlagTiers:
    def test_all_flags_have_tiers(self):
        for flag in VALID_FLAGS:
            assert flag in FLAG_TIERS

    def test_tier_values(self):
        valid_tiers = {"red", "green", "blue"}
        for tier in FLAG_TIERS.values():
            assert tier in valid_tiers

    def test_expected_red_flags(self):
        expected = {
            "small_business_set_aside",
            "lpta_source_selection",
        }
        assert expected == RED_FLAGS

    def test_expected_green_flags(self):
        expected = {"agile_methodology", "oral_presentation"}
        assert expected == GREEN_FLAGS

    def test_expected_blue_flags(self):
        expected = {
            "8a_set_aside",
            "sdvosb_set_aside",
            "hubzone_set_aside",
        }
        assert expected == BLUE_FLAGS

    def test_sets_are_disjoint(self):
        all_sets = [RED_FLAGS, GREEN_FLAGS, BLUE_FLAGS]
        for i, a in enumerate(all_sets):
            for b in all_sets[i + 1 :]:
                assert not a & b

    def test_sets_cover_all_flags(self):
        combined = RED_FLAGS | GREEN_FLAGS | BLUE_FLAGS
        assert combined == VALID_FLAGS

    def test_flag_count(self):
        assert len(VALID_FLAGS) == 7
