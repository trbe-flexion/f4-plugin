# Library Layer Implementation Plan

## Context

While training runs on SageMaker, we're building the F4 library layer (checklist step 10). This is the core public interface: `f4.filter(text)` that opp-capture will call. Designed to follow the same OEA patterns as opp-capture: Protocol-based ports, frozen dataclasses, constructor injection, adapters behind ports.

Decisions from conversation:
- FilterResult: frozen dataclass with `filter: bool`. Minimal contract — opp-capture just needs yes/no.
- Decision logic: any black flag -> filter=True. Red flag threshold (default 999 = disabled).
- FlagDetector: Protocol port for model inference. BedrockFlagDetector adapter (later). Tests use mocks.
- Taxonomy: Python dict mapping flag names to tiers, in src/domain/taxonomy.py.
- Internal logging for eval/debug mode, not exposed to caller.
- Observability (timing, token counts) logged internally, not on FilterResult.

Reference architecture: opp-capture at /Users/travisblount-elliott/Repos/flexion-opp-capture
- Ports: @runtime_checkable Protocol in domain/protocols/
- Entities: frozen dataclasses in domain/entities/
- Adapters: concrete classes in adapters/
- DI: constructor injection, composition root with build_*() factories

## Files Created

- src/domain/entities.py — FilterResult dataclass
- src/domain/taxonomy.py — FLAG_TIERS dict, VALID_FLAGS set
- src/domain/protocols.py — FlagDetector Protocol
- src/domain/parsing.py — parse model output to flag names
- src/chunking/chunker.py — text chunking with overlap
- src/decision/engine.py — flag-to-filter decision logic
- src/pipeline/filter.py — f4.filter() orchestration
- tests/test_entities.py, test_taxonomy.py, test_parsing.py, test_chunker.py, test_decision.py, test_pipeline.py

## Steps

### 1. Domain entities

src/domain/entities.py: FilterResult(filter: bool), frozen dataclass. Opp-capture gets a boolean.

### 2. Flag taxonomy

src/domain/taxonomy.py: FLAG_TIERS dict mapping flag name -> tier (black/red/green/blue). VALID_FLAGS, BLACK_FLAGS, RED_FLAGS, GREEN_FLAGS, BLUE_FLAGS as frozensets. Populated from collated-flag-set.md "Keep" section (19 flags).

### 3. Output parsing

src/domain/parsing.py: parse_flags(raw_output) -> list[str]. Strip, splitlines, filter to VALID_FLAGS. Returns empty list for "no_flag" or garbage.

### 4. Text chunking

src/chunking/chunker.py: chunk_text(text, max_tokens, overlap_tokens, tokenizer) -> list[str]. Token-boundary chunking. Tokenizer is duck-typed (encode/decode).

### 5. Decision engine

src/decision/engine.py: FilterDecisionEngine with red_flag_threshold (default 999 = disabled). Any black flag -> True. Red count >= threshold -> True.

### 6. FlagDetector protocol

src/domain/protocols.py: @runtime_checkable Protocol. detect_flags(chunk: str) -> str. Returns raw model output; parsing is separate.

### 7. Pipeline orchestration

src/pipeline/filter.py: F4Pipeline class. Constructor injection (flag_detector, tokenizer, config). filter(text) -> FilterResult. Chunks text, concurrent inference via ThreadPoolExecutor, parse/retry (1 retry then discard), deduplicate flags, apply decision engine.

### 8. Tests

50 tests, 100% coverage on src/. All model calls mocked.

## Still Pending

- BedrockFlagDetector adapter (depends on Bedrock deployment)
- RAG retrieval per chunk (depends on ChromaDB setup)
- Observability logging
