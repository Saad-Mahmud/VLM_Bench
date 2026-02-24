# vLLM MCQ Server

This folder contains a model-configured MCQ ranking server that scores options via log-likelihood (prompt logprobs), not generated-answer parsing.

## Run

```bash
python -m vllm_servers.vllm_mcq --model-alias qwen3_vl_4b --host 0.0.0.0 --port 9009
```

Supported aliases are defined in `vllm_servers/model_registry.py`.

## Health

```bash
curl http://127.0.0.1:9009/health
```

## Rank Example

```bash
curl -X POST http://127.0.0.1:9009/rank \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Look at the image and answer.\\n\\nQuestion:\\nWhich is east?\\n\\nOptions:\\nA. Georgia\\nB. North Dakota\\nC. Oklahoma\\nD. Louisiana\\n\\nCorrect Answer:",
    "options": ["Georgia", "North Dakota", "Oklahoma", "Louisiana"],
    "image_path": "datasets/mcq/science_qa/images/1554.png"
  }'
```

Response includes:

- `scores`: per-option log-likelihood scores
- `ranked`: scores sorted by descending log-likelihood
- `predicted`: top-ranked option
