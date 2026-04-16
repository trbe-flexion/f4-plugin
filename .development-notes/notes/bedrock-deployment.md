# Bedrock Deployment Log

Step-by-step record of deploying the fine-tuned Llama 3.2 3B model to Bedrock Custom Model Import.
Account: Alt/cohort (<ACCOUNT_ID>, us-east-1). Will migrate to Main later.

## 1. Merge LoRA + Export

**Run on: SageMaker** (requires GPU + model weights in ~/f4-plugin/models/adapter/)

```bash
# Clean up failed merge attempt (disk was full at 16GB EBS)
rm -rf ~/f4-plugin/models/merged

# Expanded EBS to 100GB, restarted Space

# Run merge
cd ~/f4-plugin
uv run python training/merge_and_export.py
```

Output: `models/merged/` containing model.safetensors (~6GB), config.json, generation_config.json,
tokenizer files. Script auto-fixed `tokenizer_class` from `TokenizersBackend` to `LlamaTokenizerFast`
(Bedrock requirement).

## 2. Upload to S3

**Create bucket — run on: local CLI** (authenticated to Alt account via SSO)

```bash
aws s3 mb s3://trbe-f4-finetuned-model --region us-east-1
```

**Add bucket policy — run on: local CLI** (SageMaker role lacks s3:PutObject, same as HW7)

```bash
# Get exact role ARN first:
aws iam get-role --role-name AmazonSageMakerUserIAMExecutionRole_8 --query 'Role.Arn' --output text
# Result: arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMakerUserIAMExecutionRole_8
# Note: includes service-role/ path — without it, the policy is rejected as MalformedPolicy.

aws s3api put-bucket-policy --bucket trbe-f4-finetuned-model --policy '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMakerUserIAMExecutionRole_8"},"Action":["s3:PutObject","s3:GetObject","s3:ListBucket"],"Resource":["arn:aws:s3:::trbe-f4-finetuned-model","arn:aws:s3:::trbe-f4-finetuned-model/*"]}]}'
```

**Upload model — run on: SageMaker**

```bash
cd ~/f4-plugin
aws s3 sync models/merged/ s3://trbe-f4-finetuned-model/ --region us-east-1
```

Uploaded: model.safetensors, config.json, generation_config.json, tokenizer.json,
tokenizer_config.json, chat_template.jinja.

## 3. Bedrock Custom Model Import

**Run on: AWS Console** (Bedrock → Custom Models → Import Model)

- Model name: `f4-llama-3b-flag-detector`
- S3 path: `s3://trbe-f4-finetuned-model/`
- Import Job Name: `importJob-20260316T111124`
- Service Role Name: `executionRoleName-20260316T111124` (auto-created with S3 read access)
- Region: us-east-1

Model ARN: `arn:aws:bedrock:us-east-1:165286508758:imported-model/pxi20ybyyh5t`
Status: complete

## 4. Verify Endpoint Inference

**Run on: local CLI** (Alt account, needs boto3 via uv)

```bash
uv run python scripts/test_bedrock_live.py --model-arn "arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:imported-model/3ffr95d8c4cc"
```

Results: 4/5 passed. waterfall, no_flag, COTS, small_business all correct. Agile test returned
off_the_shelf_software instead of agile_methodology — likely needs RAG context (not provided in
bare smoke test). Format compliance 100% — all responses are clean flag names, no prose.

Cold start: model threw `ModelNotReadyException` on first invocation, responded after ~1-2 min.

## 5. Chunking Without HF Tokenizer

**Decision:** Use word-based approximate chunking instead of the Llama tokenizer.

The library needs to chunk RFP text into ~512-1024 token windows for Bedrock inference. The original
design used `AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")` for exact
token-boundary chunking. This creates two problems:

1. Llama 3.2 is a gated HF repo — requires HF auth, breaking plug-and-play install
2. Bundling the tokenizer adds 16MB to the repo (full vocabulary in tokenizer.json)

Word-based chunking (~1.3 tokens/word for English prose) is close enough. Being off by 20-30 tokens
per chunk doesn't matter — the 64-token overlap exists to handle boundary imprecision. This removes
`transformers` as a runtime dependency entirely (only needed for training/eval on SageMaker).

Supersedes the tokenizer bundling approach attempted earlier (files downloaded to data/tokenizer/
but not committed).

## 6. config.json Debugging — rope_parameters vs rope_scaling

### The Problem

Model worked perfectly on SageMaker (tested at 1302 tokens, correct output) but degenerated into
gibberish ("Water You You You", repeated tokens, hallucinated text) on Bedrock for any prompt
longer than ~875 tokens. Short prompts (~221 tokens) worked fine.

Binary search (`scripts/bedrock_token_limit.py`) confirmed a hard cutoff:
- 862 tokens: PASS
- 865 tokens: FAIL

### Investigation

1. Confirmed merged model works on SageMaker at 1302 tokens — merge is not the issue
2. Confirmed RAG store content matches between training and inference — RAG is not the issue
3. Tested `messages` format (OpenAIChatCompletion) — same limit (~876 tokens)
4. Reduced `max_position_embeddings` from 131072 to 4096 — no improvement (~815 tokens)
5. Checked HW7's 1B model config — same `max_position_embeddings: 131072`, but 1B model has
   smaller KV-cache per token so the limit was never hit with short HW7 test prompts

### Root Cause

The `config.json` exported by `merge_and_export.py` (using transformers 5.5.3) used the field
name `rope_parameters`. Bedrock's inference container (documented as supporting transformers 4.51.3)
expects the older field name `rope_scaling`.

Because Bedrock couldn't read `rope_parameters`, it had no RoPE scaling configuration. Without
proper RoPE, the model's position encoding was wrong for all positions, but the error was tolerable
at short sequences and caused complete degeneration beyond ~875 tokens.

Additionally, the transformers version field (`"transformers_version": "5.5.3"`) may have caused
the inference container to misparse other config fields.

### Fix

Downloaded `config.json` from S3 and applied three changes:
1. Renamed `rope_parameters` → `rope_scaling`
2. Set `rope_scaling.factor` to `8.0` (matching Bedrock's documented override for Llama 3)
3. Moved `rope_theta: 500000.0` to top level (4.x format)
4. Changed `transformers_version` from `5.5.3` to `4.51.3`
5. Restored `max_position_embeddings` to `131072` (original value)

Re-uploaded config.json to S3, deleted old Bedrock model, re-imported.

### Result

After fix, binary search found NO failure up to 5000+ padding words (3457+ tokens). The model
now handles long prompts correctly on Bedrock.

### Lesson

**When exporting models for Bedrock Custom Model Import, the config.json must use field names
compatible with transformers 4.51.3, not newer versions.** The merge/export script runs with
whatever transformers is installed locally, which may be newer than what Bedrock supports. Key
field name change: `rope_parameters` (5.x) → `rope_scaling` (4.x).

This should be automated in `training/merge_and_export.py` alongside the existing `tokenizer_class`
fix.

## Notes

- HW7 reference bucket: `llm-class-hw7-model-trbe` (Llama 3.2 1B, still exists on Alt account)
- HW7 model ARN: `arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:imported-model/93c3bcejui1c`
- Bedrock import requires IAM role with S3 read access to the model bucket
- In HW7, SageMaker execution role lacked `s3:CreateBucket` and `s3:PutObject` — had to use
  console click-ops and manual bucket policy. Creating bucket from local CLI avoids this.
- Bedrock `invoke_model` API: `{"prompt": formatted, "max_gen_len": N, "temperature": T, "top_p": P}`
  → response: `{"generation": "..."}`
- Bedrock also supports `messages` format (OpenAIChatCompletion): `{"messages": [...], "max_tokens": N}`
  → response: OpenAI-compatible JSON with `choices[0].message.content`
