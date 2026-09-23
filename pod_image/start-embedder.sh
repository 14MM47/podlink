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

# Keep TEI's startup INFO line out of the logs: it prints its full argument list,
# INCLUDING api_key in plain text (seen in RunPod container logs 2026-09-23; only
# the HF token is masked). /health gates readiness, not log lines, so warn-level
# logging loses nothing podlink uses. TEI reads LOG_LEVEL (NOT RUST_LOG — verified
# against the TEI 1.9 router: RUST_LOG=warn still printed the key, LOG_LEVEL=warn
# did not). An explicit LOG_LEVEL still wins.
export LOG_LEVEL="${LOG_LEVEL:-warn}"

# TEI warms up with a full --max-batch-tokens batch (default 16384) on top of the model
# weights. Beside a vLLM that already holds its GPU share that warm-up OOMs (seen 16 Sept
# 2026: 8.4 GiB weights loaded, then CUDA_ERROR_OUT_OF_MEMORY at warm-up with ~10 GB
# free). RAG chunks are far shorter, so a 4096-token ceiling costs nothing and warms up in
# a quarter of the activation memory. Override with EMBED_MAX_BATCH_TOKENS.
exec text-embeddings-router \
  --model-id "${EMBED_MODEL_ID}" \
  --hostname 0.0.0.0 \
  --port 8080 \
  --max-batch-tokens "${EMBED_MAX_BATCH_TOKENS:-4096}"
