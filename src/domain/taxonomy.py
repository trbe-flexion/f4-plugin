"""Flag taxonomy for F4.

Maps flag names to tiers. Source of truth is collated-flag-set.md.
Only includes flags assigned to the small LLM (the "Keep" set).
"""

from __future__ import annotations

FLAG_TIERS: dict[str, str] = {
    # Black (fast-fail)
    "waterfall_methodology": "black",
    "off_the_shelf_software": "black",
    "no_custom_development": "black",
    "onsite_required": "black",
    "budget_too_low": "black",
    # Red
    "small_business_set_aside": "red",
    "brownfield": "red",
    "lpta_source_selection": "red",
    "marginal_short_duration": "red",
    # Blue (informational)
    "large_team": "blue",
    "8a_set_aside": "blue",
    "wosb_set_aside": "blue",
    "edwosb_set_aside": "blue",
    "sdvosb_set_aside": "blue",
    "hubzone_set_aside": "blue",
    "onsite_madison": "blue",
    # Green
    "agile_methodology": "green",
    "oral_presentation": "green",
    "design_exercise": "green",
}

VALID_FLAGS: frozenset[str] = frozenset(FLAG_TIERS.keys())

BLACK_FLAGS: frozenset[str] = frozenset(
    name for name, tier in FLAG_TIERS.items() if tier == "black"
)

RED_FLAGS: frozenset[str] = frozenset(name for name, tier in FLAG_TIERS.items() if tier == "red")

GREEN_FLAGS: frozenset[str] = frozenset(
    name for name, tier in FLAG_TIERS.items() if tier == "green"
)

BLUE_FLAGS: frozenset[str] = frozenset(name for name, tier in FLAG_TIERS.items() if tier == "blue")
