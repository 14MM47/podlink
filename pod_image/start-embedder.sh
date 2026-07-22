#!/usr/bin/env bash
# Launch the TEI embedder on :8080. TEI >=1.5 exposes OpenAI /v1/embeddings and a
# /health endpoint (which podlink gates readiness on). TEI is keyless internally.
# The HF token (for gated/rate-limited weight pulls) is inherited from the
# container env ($HF_TOKEN / $HUGGING_FACE_HUB_TOKEN), injected by pod_up.
set -euo pipefail

# Gate the endpoint: this port is reachable over RunPod's PUBLIC HTTPS proxy, so
# it must NOT be keyless. TEI reads API_KEY from env natively, so we export it
# here rather than passing --api-key on argv (which would be visible in `ps aux`
# via the RunPod web terminal). podlink injects TEI_API_KEY; unset => fail closed.
export API_KEY="${TEI_API_KEY}"

exec text-embeddings-router \
  --model-id "${EMBED_MODEL_ID}" \
  --hostname 0.0.0.0 \
  --port 8080
