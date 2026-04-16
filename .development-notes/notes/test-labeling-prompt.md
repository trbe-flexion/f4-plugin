# Test Set Labeling Prompt

This is the prompt sent to Opus for each chunk in `scripts/label_test_set.py`.
`{flag_block}` is expanded from the FLAG_DESCRIPTIONS dict. `{chunk_text}` is a ~400-word chunk.

---

Your output will be parsed directly by a script. You MUST return ONLY
a JSON object in this exact format, with no other text:
{"flags": ["flag_name_1", "flag_name_2"]}
or if no flags: {"flags": []}

## Context

You are labeling government RFP text chunks for Flexion, a software consultancy
that screens RFPs to decide whether to bid. Each flag represents a specific
business signal that affects the bid/no-bid decision. These labels are ground
truth for evaluating a fine-tuned detection model — accuracy matters more than
coverage. A false positive is worse than a missed flag.

You are seeing one ~400-word chunk at a time, not the full document. You must
decide based ONLY on what is explicitly stated in this chunk. Do not infer flags
from context clues, document titles, boilerplate, or indirect language. Most
chunks contain no flags — that is expected and correct.

## Flags to detect:

  - off_the_shelf_software: The chunk describes procuring, configuring, or deploying
    a commercial off-the-shelf product as the PRIMARY work. Must contain explicit COTS
    language IN THE CHUNK BODY: "off-the-shelf," "COTS," "GOTS," "NDI," "FAR Part 12,"
    "shrink-wrapped," "OOTB," "out-of-the-box." A document title mentioning a platform
    name is NOT sufficient. Generic procurement clauses are NOT sufficient.
  - lpta_source_selection: Source selection is Lowest Price Technically Acceptable.
    Look for: "LPTA," "lowest price technically acceptable."
  - small_business_set_aside: RFP is set aside exclusively for small businesses.
    Look for FAR 52.219-6, "small business set-aside," or explicit restriction.
  - 8a_set_aside: 8(a) Business Development Program set-aside. Look for
    FAR 52.219-11/14, "8(a) sole source," "8(a) competitive set-aside."
  - wosb_set_aside: Women-Owned Small Business (WOSB) or EDWOSB set-aside.
    Look for FAR 52.219-29/30, WOSB/EDWOSB language.
  - sdvosb_set_aside: Service-Disabled Veteran-Owned Small Business set-aside.
    Look for FAR 52.219-27, SDVOSB language.
  - hubzone_set_aside: HUBZone set-aside. Look for FAR 52.219-13, HUBZone price
    evaluation preference, or explicit HUBZone set-aside language.
  - agile_methodology: The chunk explicitly requires or describes Agile/Scrum
    methodology for the work. Look for: "Agile," "Scrum," "sprint," "user stories,"
    "iterative development." General mentions of modern practices or CI/CD alone
    are NOT sufficient — the chunk must specifically reference Agile as a methodology.
  - oral_presentation: The chunk describes an oral presentation as part of the
    evaluation/award process. Look for: "oral presentation," "oral proposal."
  - design_exercise: The chunk describes a design challenge, prototype, proof of
    concept, or live demonstration required as part of evaluation. Look for:
    "demonstration," "challenge scenario," "proof of concept," "POC," "flyoff," "pilot."
  - budget_too_low: The chunk states a total contract value, ceiling, or NTE under
    $100K as an explicit dollar figure.
  - onsite_required: The chunk explicitly requires all work at a specific location.
    Look for: "on-site," "onsite," "place of performance," "in-person," "physically
    present." Do NOT flag if remote, hybrid, or telework options are mentioned.
  - large_team: The chunk explicitly states 10+ personnel/FTEs or enumerates 10+
    named roles. Must be an explicit count or list, not inferred from scope size.
  - marginal_short_duration: The chunk explicitly states a period of performance
    under 12 months. Must be a stated duration, not inferred.

## Rules:
- Only flag what is EXPLICITLY stated in the chunk text
- When in doubt, return no flags — precision over recall
- Boilerplate, contract clauses, and legal terms are almost never flaggable
- Most chunks will have no flags. That is correct.

## Chunk:
{chunk_text}

Return ONLY: {"flags": ["flag_name", ...]} or {"flags": []}
