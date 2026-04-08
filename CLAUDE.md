# CLAUDE.md

## Code Standards

- All files must end with a newline character (enforced by pre-commit hook).
- Always run BOTH `ruff check .` and `ruff format .` before committing. `ruff format` does NOT catch linting errors.
- Common ruff issues: bare `except:` (use `except Exception:`), f-strings without placeholders.
- Transformers API: use `dtype` parameter, NOT `torch_dtype` (deprecated).
  - Correct: `AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16)`
- Testing: pytest with 80%+ coverage target. Mock model calls for CI performance. Run with `pytest --cov`.
- Package management: uv (`uv sync`, `uv add`, etc.)

## Lessons from Prior Assignments

- SageMaker: Virtual environments must be on local disk, not the S3-mounted home directory. Pip/uv installs fail silently or corrupt on S3 FUSE mounts.
- GPU sizing: Qwen2.5-32B OOM'd on an L4 (24GB VRAM). Qwen2.5-7B in float16 fits comfortably (~14GB). For LoRA fine-tuning, expect higher VRAM than inference.
- Bedrock Converse API: HW7's `BedrockInferenceClient` (archived at `archive/hw7/`) is a working reference for calling Bedrock models. The Converse API is model-agnostic.
- ChromaDB: HW8's `vectordb.py` (archived at `archive/hw8/`) has a working embedding + retrieval implementation.
- TextGrad: HW10 has working TextGrad code in `personal_notes/HW10/pgp_*.py`. The backward engine (critic) should be more capable than the forward engine.
- Gradio: HW11 used Gradio Blocks (not ChatInterface) wrapping Bedrock Converse. Working reference at `archive/hw11/` or `src/frontend.py`.
- Outlines: HW9 used Outlines CFG mode for constrained HTML generation. Not applicable to Bedrock but useful for local dev/eval on SageMaker.

## Writing Style

See `STYLE_GUIDE.md` for ADR and documentation conventions.
