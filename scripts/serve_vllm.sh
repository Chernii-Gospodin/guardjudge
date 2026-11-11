#!/usr/bin/env bash
# Serve the fine-tuned judge with vLLM behind an OpenAI-compatible API on :8001 (Linux + GPU).
# The full fine-tuned checkpoint is a regular HF model: no adapter merge is needed.
# Lighter alternative without vLLM (CPU / Windows): python scripts/serve.py
# DTYPE: float16 on T4/P100 (default), bfloat16 on Ampere+.
# MAX_LEN: keep at/under the training max_length.
MODEL_DIR="${1:-outputs/guardjudge-qwen3-0.6b}"
DTYPE="${DTYPE:-float16}"
MAX_LEN="${MAX_LEN:-2048}"
vllm serve "$MODEL_DIR" --served-model-name guardjudge --port 8001 \
  --dtype "$DTYPE" --max-model-len "$MAX_LEN"
