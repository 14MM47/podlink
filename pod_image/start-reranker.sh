#!/usr/bin/env bash
# Launch the TEI reranker on :8081. TEI auto-detects a cross-encoder / sequence-
# classification model and exposes /rerank (the client POSTs {query, texts} to it)
# plus /health (podlink gates readiness on it). Keyless internally.
set -euo pipefail

# Gate the endpoint: this port is reachable over RunPod's PUBLIC HTTPS proxy, so
# it must NOT be keyless. TEI reads API_KEY from env natively, so we export it
# here rather than passing --api-key on argv (visible in `ps aux`). podlink
# injects TEI_API_KEY; unset => fail closed (service won't start without a key).
export API_KEY="${TEI_API_KEY}"

exec text-embeddings-router \
  --model-id "${RERANK_MODEL_ID}" \
  --hostname 0.0.0.0 \
  --port 8081
