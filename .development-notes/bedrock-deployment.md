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
- Import Job Name: `importJob-20260311T220627`
- Service Role Name: `executionRoleName-20260311T220627` (auto-created with S3 read access)
- Region: us-east-1

Model ARN: `arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:imported-model/hzlmk7msk3dn`
Status: complete

## 4. Verify Endpoint Inference

**Run on: local CLI** (Alt account, needs boto3 via uv)

```bash
uv run python scripts/test_bedrock_live.py --model-arn "arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:imported-model/hzlmk7msk3dn"
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

## Notes

- HW7 reference bucket: `llm-class-hw7-model-trbe` (Llama 3.2 1B, still exists on Alt account)
- HW7 model ARN: `arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:imported-model/93c3bcejui1c`
- Bedrock import requires IAM role with S3 read access to the model bucket
- In HW7, SageMaker execution role lacked `s3:CreateBucket` and `s3:PutObject` — had to use
  console click-ops and manual bucket policy. Creating bucket from local CLI avoids this.
- Bedrock `invoke_model` API: `{"prompt": formatted, "max_gen_len": N, "temperature": T, "top_p": P}`
  → response: `{"generation": "..."}`
