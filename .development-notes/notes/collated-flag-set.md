F4 screens government RFPs on behalf of Flexion, a software development consultancy specializing in Agile delivery of
  complex, custom software systems for government clients. The goal is to identify RFPs that are a poor fit before
  expensive analysis occurs.

  ---
  Tiers

  - Black (fast-fail): Immediately disqualify. Do not pass to further evaluation. Only fire when the evidence in the text
  is explicit and unambiguous. A false positive here means a real revenue opportunity is permanently discarded.
  - Red: Concern that reduces attractiveness but does not disqualify. Pass to further evaluation. Multiple red flags may
  combine to disqualify depending on harness configuration.
  - Blue: Neutral informational context. Notable but neither positive nor negative on its own.
  - Green: Positive signal. Increases attractiveness.

  ---
  Black Flags

  - scope_misalignment — The primary effort described in the RFP is clearly not custom software development. Examples:
  management consulting, policy research, strategy advisory. Only fire on explicit summary-level scope statements. Do not
  fire because a chunk discusses non-software topics incidentally within an otherwise software-focused RFP.
  - waterfall_methodology — The RFP explicitly requires sequential/waterfall development. Look for: "waterfall,"
  "traditional SDLC," "sequential phases," "standard software lifecycle development," "fixed requirements baseline," "phase
   gate," "V-model." Flexion is an Agile shop — waterfall is incompatible.
  - onsite_required — All work must be performed at a specific location. Exception: Madison, WI is acceptable. Do not fire
  if hybrid, remote, or flexible options are offered.
  - narrow_scope — Work is limited to a single non-development discipline (UX only, PM only, QA only) with no custom
  software product being built. Flexion delivers full software products, not staff augmentation in a single specialty.
  - off_the_shelf_software — Primary work is configuring or deploying commercial off-the-shelf platforms rather than
  building custom software. Look for specific product names (Salesforce, SharePoint, WordPress, Wix, Squarespace) or COTS
  language ("off-the-shelf," "commercial solution," "configure and deploy," "evaluate and deploy commercial solutions,"
  "pre-built platform").
  - no_custom_development — The RFP explicitly excludes custom software development. Look for language in scope or
  exclusions: "no custom development," "no software development," "configuration only," "COTS solution required."
  - hardware_procurement — The primary deliverable is hardware supply or infrastructure procurement, not software. Look
  for: "hardware procurement," "procure and install hardware," "server hardware," "network equipment." Do not fire because
  hardware is mentioned in a software RFP context.
  - insufficient_timeline — Proposal deadline is less than 14 days away. This is typically a metadata check, but may appear
   in text.
  - budget_too_low — Total contract budget is below $100K. May appear in text as a dollar figure.

  ---
  Red Flags

  - small_business_set_aside — RFP is set aside exclusively for small businesses. Flexion does not qualify as a small
  business and would need to partner to bid, which significantly reduces attractiveness.
  - lpta_source_selection — Source selection is Lowest Price Technically Acceptable. Flexion competes on value and
  expertise, not price. Look for: "LPTA," "lowest price technically acceptable."
  - feature_factory — RFP emphasizes feature delivery volume without discovery, user research, or iterative learning.
  Suggests a client who wants output, not outcomes. Flexion's Agile approach is misaligned with pure output-driven
  contracts.
  - marginal_investment — Not directly detectable. Derived by the harness from sub-criteria below.
  - marginal_short_duration — Period of performance is less than 12 months.
  - marginal_low_budget — Total contract value is under $1M.
  - marginal_small_team — Scope requires fewer than 3 FTEs.
  - marginal_uncertain_funding — Funding is contingent on future appropriations or lacks a clear commitment.

  ---
  Blue Flags (Informational)

  - tight_deadline — Proposal due within 2 weeks. Informational — not a disqualifier but relevant context.
  - large_team — Scope requires 10+ people. Notable for capacity planning.
  - 8a_set_aside — 8(a) Business Development Program set-aside. Flexion may have a JV partner that qualifies.
  - wosb_set_aside — Women-Owned Small Business set-aside.
  - edwosb_set_aside — Economically Disadvantaged Women-Owned Small Business set-aside.
  - sdvosb_set_aside — Service-Disabled Veteran-Owned Small Business set-aside.
  - hubzone_set_aside — HUBZone set-aside.
  - onsite_madison — Onsite work is required, but the specified location is Madison, WI. Flexion is headquartered there;
  this negates the onsite_required black flag.

  ---
  Green Flags

  - oral_presentation — RFP includes an oral presentation component. Flexion performs strongly in oral presentations; this
  differentiates from price-only evaluation.
  - design_exercise — RFP includes a design challenge, prototype, or exercise. Demonstrates Flexion's capabilities
  directly.
  - agile_methodology — RFP explicitly requires or expects Agile/Scrum methodology. Direct alignment with Flexion's
  delivery model.

  ---
  Keep (Small LLM on Chunks)

  Flags with explicit, formulaic language that a fine-tuned small model can reliably detect from a partial chunk.
  Reduced to 7 high-performing flags after evaluation on real RFP data:

  - lpta_source_selection
  - small_business_set_aside
  - 8a_set_aside
  - sdvosb_set_aside
  - hubzone_set_aside
  - agile_methodology
  - oral_presentation
  - no_flag (special: output when no flags are detected in a chunk; filtered out by harness, not a real flag)

  ---
  Drop (Not assigned to small LLM)

  Previously in Keep, dropped because chunk-level context is insufficient:
  - budget_too_low — Dollar figure in a chunk can't be confirmed as total contract value.
  - onsite_required — Nuance around hybrid/remote exceptions requires broader document context.
  - large_team — FTE count in a chunk is typically a subteam, not full scope.

  Previously in Keep, dropped due to poor model performance (likely needs better training data):
  - off_the_shelf_software — High false-positive rate on real data.
  - wosb_set_aside — Insufficient training signal (includes edwosb; distinction too narrow).
  - design_exercise — Insufficient training signal.
  - marginal_short_duration — Insufficient training signal.

  Never assigned to small LLM:
  - scope_misalignment — Requires document-level context to distinguish incidental non-software content from a genuinely
  out-of-scope RFP.
  - narrow_scope — Needs holistic read to confirm no software product is being built.
  - hardware_procurement — Easily false-pops on hardware mentions within software RFPs.
  - insufficient_timeline — Metadata check; deadline date rarely appears in body text chunks.
  - feature_factory — Requires judgment about overall delivery philosophy, not detectable from a chunk.
  - marginal_low_budget — Dollar figure in a chunk can't be confirmed as total contract value.
  - marginal_small_team — FTE count in a chunk is typically a subteam, not full scope.
  - marginal_uncertain_funding — Appropriations boilerplate appears in nearly every federal contract; high false-pop risk.
  - marginal_investment — Derived flag; computed by harness from sub-criteria, not model-detected.
  - no_custom_development — Requires document-level context; "no custom dev" is rarely stated explicitly in a single chunk.
  Sonnet mislabeled product purchases as this flag in ~92% of cases.
  - waterfall_methodology — 0 validated examples from 958 RFPs. Genuinely rare in the corpus; downstream LLM will catch it.
  - onsite_madison — When onsite_required fires, opp-capture can string-match for "Madison, WI" to negate it.
  Not implemented in f4-plugin; documented here for the integration layer.
  - brownfield — No consistent chunk-level signal. Language is too diffuse ("maintain," "transition," "modernize" each
  appear in <35% of examples). Requires document-level understanding of whether contractor inherits existing code.
  - edwosb_set_aside — Merged into wosb_set_aside for training.

  ---
  All Flags by Color

  Black:
  - budget_too_low
  - hardware_procurement
  - insufficient_timeline
  - narrow_scope
  - no_custom_development
  - off_the_shelf_software
  - onsite_required
  - scope_misalignment
  - waterfall_methodology

  Red:
  - feature_factory
  - lpta_source_selection *
  - marginal_investment
  - marginal_low_budget
  - marginal_short_duration
  - marginal_small_team
  - marginal_uncertain_funding
  - small_business_set_aside *

  Blue:
  - 8a_set_aside *
  - edwosb_set_aside
  - hubzone_set_aside *
  - large_team
  - onsite_madison
  - sdvosb_set_aside *
  - tight_deadline
  - wosb_set_aside

  Green:
  - agile_methodology *
  - design_exercise
  - oral_presentation *
