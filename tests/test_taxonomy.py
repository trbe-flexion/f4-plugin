from src.domain.taxonomy import (
    BLACK_FLAGS,
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
        valid_tiers = {"black", "red", "green", "blue"}
        for tier in FLAG_TIERS.values():
            assert tier in valid_tiers

    def test_expected_black_flags(self):
        expected = {
            "waterfall_methodology",
            "off_the_shelf_software",
            "no_custom_development",
            "onsite_required",
            "budget_too_low",
        }
        assert expected == BLACK_FLAGS

    def test_expected_red_flags(self):
        expected = {
            "small_business_set_aside",
            "brownfield",
            "lpta_source_selection",
            "marginal_short_duration",
        }
        assert expected == RED_FLAGS

    def test_expected_green_flags(self):
        expected = {"agile_methodology", "oral_presentation", "design_exercise"}
        assert expected == GREEN_FLAGS

    def test_expected_blue_flags(self):
        expected = {
            "large_team",
            "8a_set_aside",
            "wosb_set_aside",
            "edwosb_set_aside",
            "sdvosb_set_aside",
            "hubzone_set_aside",
            "onsite_madison",
        }
        assert expected == BLUE_FLAGS

    def test_sets_are_disjoint(self):
        all_sets = [BLACK_FLAGS, RED_FLAGS, GREEN_FLAGS, BLUE_FLAGS]
        for i, a in enumerate(all_sets):
            for b in all_sets[i + 1 :]:
                assert not a & b

    def test_sets_cover_all_flags(self):
        combined = BLACK_FLAGS | RED_FLAGS | GREEN_FLAGS | BLUE_FLAGS
        assert combined == VALID_FLAGS

    def test_flag_count(self):
        assert len(VALID_FLAGS) == 19
