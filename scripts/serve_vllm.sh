#!/usr/bin/env bash
# Serve the merged judge behind an OpenAI-compatible API on :8001.
# DTYPE: float16 on T4/P100 (default), bfloat16 on Ampere+.
# MAX_LEN: keep at/under the training max_length.
# TRUST_REMOTE_CODE=1 only for model repos you fully trust (runs their bundled Python).
MODEL_DIR="${1:-merged/guardjudge-4b}"
DTYPE="${DTYPE:-float16}"
MAX_LEN="${MAX_LEN:-2048}"
REMOTE_CODE="${TRUST_REMOTE_CODE:+--trust-remote-code}"
vllm serve "$MODEL_DIR" --served-model-name guardjudge --port 8001 \
  --dtype "$DTYPE" --max-model-len "$MAX_LEN" ${REMOTE_CODE}
