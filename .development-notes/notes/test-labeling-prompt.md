# Labeling Prompts

Both prompts used in `scripts/relabel_training_data.py`.
`{flag_block}` is expanded from the FLAG_DESCRIPTIONS dict. `{chunk}` is a ~400-word chunk.

**JSON enforcement**: All API calls use assistant prefill — the assistant message
starts with `{"flags":` so the model can only complete the JSON array. This
prevents reasoning preamble from breaking the JSON parser.

---

## Pass 1: Labeling Prompt (`build_labeling_prompt`)

Your output will be parsed directly by a script. You MUST return ONLY
a JSON object in this exact format, with no other text:
{"flags": ["flag_name_1", "flag_name_2"]}
or if no flags: {"flags": []}

### Context

You are labeling government RFP text chunks for Flexion, a software consultancy
that screens RFPs to decide whether to bid. Each flag represents a specific
business signal that affects the bid/no-bid decision. These labels are ground
truth for a fine-tuned detection model that replaces simple string matching.
The model must be smarter than keyword search in both directions: catching real
signals that string matching would miss, and ignoring boilerplate that string
matching would falsely trigger on. Flag with confidence when the evidence is
clear, but do not flag on weak or ambiguous evidence.

You are seeing one ~400-word chunk at a time, not the full document. You must
decide based on what is present in this chunk, not inferred from context clues,
document titles, boilerplate, or indirect language.

### Flags to detect:

  - scope_misalignment: This chunk is a top-level scope statement — an introduction,
    overview, or purpose section that describes what the entire contract is for. The
    work described is clearly not custom software development. Do not flag task-level
    details, work breakdown items, technical requirements, or operational procedures,
    even if they describe non-software work. Only the chunk that answers "what is this
    contract for?" should be flagged. Administrative sections, evaluation criteria,
    CLINs, and other non-scope content must never trigger this flag. A false positive
    permanently discards a revenue opportunity — only flag when confidence is very
    high.
  - waterfall_methodology: The chunk explicitly requires sequential/waterfall
    development as the methodology. Look for: "waterfall," "traditional SDLC,"
    "sequential phases," "fixed requirements baseline," "phase gate," "V-model."
    Project phases, contract phases, and hardware installation sequences do NOT
    count — only actual waterfall SDLC mandates.
  - off_the_shelf_software: This chunk explicitly states what the procurement is for,
    and the primary deliverable is acquiring, licensing, or standing up a commercial
    off-the-shelf product — not building custom software. Ongoing support,
    maintenance, administration, or operations of commercial products already in use
    does not qualify. Only flag chunks that directly describe the overall purpose or
    scope of the contract. Administrative sections, task details, evaluation criteria,
    and other non-summary content must never trigger this flag, regardless of what
    they imply about the RFP. A false positive permanently discards a revenue
    opportunity — only flag when confidence is very high.
  - lpta_source_selection: Source selection for THIS solicitation is Lowest Price
    Technically Acceptable. Look for: "LPTA," "lowest price technically acceptable."
    The LPTA must apply to the evaluation of this RFP — do NOT flag references to
    LPTA for future task orders, add-ons, options, or other solicitations.
  - small_business_set_aside: This solicitation is set aside exclusively for small
    businesses. Look for: "small business set-aside," "total small business set-aside,"
    FAR 52.219-6. The set-aside must be stated as applying to THIS solicitation — a
    FAR clause listed in an incorporated-by-reference table is NOT sufficient.
  - 8a_set_aside: This solicitation is set aside under the 8(a) Business Development
    Program. Look for: "8(a) set-aside," "8(a) sole source," "8(a) competitive,"
    FAR 52.219-11/14. Must apply to THIS solicitation, not just listed in boilerplate.
  - wosb_set_aside: This solicitation is set aside for Women-Owned Small Businesses
    (WOSB) or Economically Disadvantaged WOSB (EDWOSB). Look for: "WOSB set-aside,"
    "EDWOSB," FAR 52.219-29/30. Must apply to THIS solicitation, not just listed
    in boilerplate.
  - sdvosb_set_aside: This solicitation is set aside for Service-Disabled
    Veteran-Owned Small Businesses. Look for: "SDVOSB set-aside," "service-disabled
    veteran," FAR 52.219-27. Must apply to THIS solicitation, not just listed in
    boilerplate.
  - hubzone_set_aside: This solicitation is set aside for HUBZone businesses. Look
    for: "HUBZone set-aside," "HUBZone price evaluation preference," FAR 52.219-13.
    Must apply to THIS solicitation, not just listed in boilerplate.
  - agile_methodology: The chunk explicitly requires or describes Agile/Scrum
    methodology for the work. Look for: "Agile," "Scrum," "sprint," "user stories,"
    "iterative development." General mentions of modern practices or CI/CD alone
    are NOT sufficient — the chunk must specifically reference Agile as a methodology.
  - oral_presentation: The chunk describes a structured oral presentation or oral
    proposal as a distinct step in the evaluation/award process. Look for: "oral
    presentation," "oral proposal," "offerors shall present." The chunk must describe
    the oral presentation as an evaluation event — passing mentions of presentations
    in general context, interview logistics, or past performance narratives do NOT
    qualify. Flag even if the oral presentation is indicated to be optional.
  - design_exercise: The chunk describes a design challenge, prototype, proof of
    concept, or live demonstration as a distinct step in the evaluation/award process.
    Look for: "design challenge," "challenge scenario," "proof of concept," "POC,"
    "flyoff," "pilot," "demonstration scenario." The chunk must describe the exercise
    as an evaluation event — general mentions of prototyping or demos in the context
    of delivery methodology do NOT qualify. Flag even if the design exercise is
    indicated to be optional.
  - budget_too_low: The chunk states a total contract value, ceiling, or NTE under
    $100K as an explicit dollar figure. The dollar amount must clearly represent the
    overall monetary allocation for the entire contract (e.g., "total contract value,"
    "ceiling price," "not to exceed," "total estimated cost"). Do NOT flag: CLIN line
    items, individual option year values, travel budgets, subcontract amounts, funding
    increments, per-unit or hourly rates, dollar thresholds in policy/reporting rules,
    or other partial figures. When in doubt, do not flag.
    A false positive permanently discards a revenue opportunity — only flag when
    confidence is very high.
  - onsite_required: The chunk explicitly mandates that all or substantially all work
    must be performed at a specific physical location. Look for: "on-site," "onsite,"
    "in-person," "physically present," "work shall be performed at." A "place of
    performance" section header or address alone is NOT sufficient — the chunk must
    state that onsite presence is required as a condition of the work. Do NOT flag if
    remote, hybrid, or telework options are mentioned. Do NOT flag if the location is
    Madison, WI (use onsite_madison instead).
    A false positive permanently discards a revenue opportunity — only flag when
    confidence is very high.
  - onsite_madison: The chunk explicitly mandates onsite work AND specifies Madison,
    WI (or Madison, Wisconsin) as the location. Both conditions must be present in
    the chunk: an explicit onsite mandate + Madison, WI. A "place of performance"
    header or address alone is NOT sufficient — the chunk must state that onsite
    presence is required. If onsite is required but the location is not Madison, use
    onsite_required instead.
  - brownfield: The contractor will inherit and work within an existing codebase
    developed by a prior team. The chunk must make clear that the contractor is
    expected to take over, maintain, or extend running software — not build a
    replacement. Generic terms like "modernize," "transition," or "legacy" are
    NOT sufficient on their own, as they appear frequently in non-brownfield
    contexts. The chunk must contain strong, specific evidence that existing code
    is being handed to the contractor. When in doubt, do not flag.
  - large_team: The chunk indicates the contractor's team must include 10 or more
    personnel or FTEs. Must be an explicit count of the contractor's own team — not
    user counts, end-user populations, vendor counts, government staff, or
    site/location counts. The number must clearly refer to the contractor's staffing
    requirement. When in doubt, do not flag.
  - marginal_short_duration: The total period of performance for the contract is
    under 12 months. Must be a stated total duration, not an individual option period,
    task order duration, phase duration, or transition period within a longer contract.
    12 months exactly does NOT qualify. When in doubt, do not flag.

### Rules:
- Only flag what is present in the chunk text, not inferred from indirect clues or boilerplate
- Consider each flag independently. For each flag, ask: does this chunk contain
  direct evidence for this specific flag? Do not let the presence of one flag
  influence your judgment of others.
- When in doubt about a specific flag, do not apply it
- Boilerplate, contract clauses, and legal terms are almost never flaggable
- It is perfectly acceptable for a chunk to have no flags
- Your output MUST return ONLY a JSON object in this exact format, with no other text:
{"flags": ["flag_name_1", "flag_name_2"]}
or if no flags: {"flags": []}

### Chunk:
{chunk}

Return ONLY: {"flags": ["flag_name", ...]} or {"flags": []}

---

## Pass 2: Validation Prompt (`build_validation_prompt`)

`{flags_str}` is the comma-separated list of flags from Pass 1.
`{flag_block}` shows only the definitions for the flagged flags, with BLACK flags
prefixed by `[BLACK — fast-fail]`.

BLACK_FLAGS = {scope_misalignment, onsite_required, off_the_shelf_software,
budget_too_low}

CRITICAL: Return ONLY a JSON object. No reasoning, no explanation, no
preamble. Your entire response must be parseable as JSON.

Format: {"flags": ["flag_name_1"]} or {"flags": []}

---

Validate these flags against the chunk below. Remove any flag that lacks
clear, direct textual evidence per its definition. Keep only flags with
explicit support in the chunk text.

Flags to validate: {flags_str}

Definitions:
{flag_block}

Removal criteria (apply strictly):
- No specific text in the chunk satisfies the definition -> remove
- BLACK flags are fast-fail (auto-disqualify the RFP). False positives
  permanently discard revenue opportunities. Remove unless unambiguous.
- Evidence is in an administrative section (eval criteria, CLINs, contract
  clauses, SF 1449 forms, FAR provisions, continuation sheets) -> remove
- Evidence is indirect, boilerplate, headers, or context clues -> remove
- You may ONLY keep flags from the original list -- do not add new flags

Chunk:
{chunk}

Respond with ONLY: {"flags": [...]}
