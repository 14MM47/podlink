#!/usr/bin/env bash
# Launch the reranker on :8081. Two interchangeable backends, picked by
# RERANK_BACKEND (default tei), both answering /health so podlink's readiness gate
# and health tiles are identical either way:
#
#   tei  — TEI auto-detects a cross-encoder / sequence-classification model and
#          exposes native /rerank (client POSTs {query, texts}). Right for
#          bge-reranker-v2-m3. TEI cannot serve LLM-based rerankers (Qwen3-Reranker,
#          zerank-2) as of 1.9.4.
#   vllm — a second vLLM process with the pooling runner, exposing Cohere-style
#          /v1/rerank (client POSTs {query, documents}). Right for LLM-based
#          rerankers. It binds 127.0.0.1:18081; the public :8081 is then the
#          authproxy (nginx), because vLLM's --api-key only guards /v1, /v2 and
#          /inference — its root /rerank, /score, /pooling would be open.
set -euo pipefail

BACKEND="${RERANK_BACKEND:-tei}"

if [[ "${BACKEND}" == "tei" ]]; then
  # Gate the endpoint: this port is reachable over RunPod's PUBLIC HTTPS proxy, so
  # it must NOT be keyless. TEI reads API_KEY from env natively, so we export it
  # here rather than passing --api-key on argv (visible in `ps aux`). podlink
  # injects TEI_API_KEY; unset => fail closed (service won't start without a key).
  export API_KEY="${TEI_API_KEY}"
  # Keep TEI's startup INFO line out of the logs: it prints its full argument list,
  # INCLUDING api_key in plain text (seen in RunPod container logs 2026-09-23; only
  # the HF token is masked). /health gates readiness, not log lines, so warn-level
  # logging loses nothing podlink uses. TEI reads LOG_LEVEL (NOT RUST_LOG — verified
  # against the TEI 1.9 router: RUST_LOG=warn still printed the key, LOG_LEVEL=warn
  # did not). An explicit LOG_LEVEL still wins.
  export LOG_LEVEL="${LOG_LEVEL:-warn}"

  exec text-embeddings-router \
    --model-id "${RERANK_MODEL_ID}" \
    --hostname 0.0.0.0 \
    --port 8081
fi

if [[ "${BACKEND}" != "vllm" ]]; then
  # Unknown value: fail loudly (supervisor retries, the health tile stays red)
  # rather than silently serving the wrong backend.
  echo "[start-reranker] ERROR: RERANK_BACKEND must be 'tei' or 'vllm', got '${BACKEND}'" >&2
  exit 1
fi

# ---- vllm backend ------------------------------------------------------------
# Fail closed on a missing key: vLLM with no key would serve /v1/rerank keyless on
# the public proxy. The key itself is read by vLLM from VLLM_API_KEY (env, not argv).
if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "[start-reranker] ERROR: VLLM_API_KEY unset; refusing to start a keyless reranker" >&2
  exit 1
fi

# Wait for the TEI embedder to be healthy before this vLLM profiles GPU memory:
# vLLM's startup memory profile aborts if another process on the GPU RELEASES
# memory mid-profile, which TEI's warm-up does. (start-vllm.sh waits for BOTH
# :8080 and :8081 for the same reason, so the LLM still starts last.)
wait_s="${RERANK_WAIT_FOR_EMBEDDER_S:-1200}"
# Integer seconds only: a typo like "off" would otherwise compare as 0 and skip
# the wait this exists for. Fall back to the default and say so.
if ! [[ "${wait_s}" =~ ^[0-9]+$ ]]; then
  echo "[start-reranker] WARNING: RERANK_WAIT_FOR_EMBEDDER_S='${wait_s}' is not integer seconds; using 1200" >&2
  wait_s=1200
fi
if [[ "${wait_s}" != "0" ]]; then
  auth=()
  if [[ -n "${TEI_API_KEY:-}" ]]; then auth=(-H "Authorization: Bearer ${TEI_API_KEY}"); fi
  started=$(date +%s)
  until curl -fsS -m 5 "${auth[@]}" "http://127.0.0.1:8080/health" >/dev/null 2>&1; do
    if (( $(date +%s) - started >= wait_s )); then
      echo "[start-reranker] WARNING: embedder not healthy after ${wait_s}s; starting vLLM reranker anyway" >&2
      break
    fi
    sleep 10
  done
fi

# Model presets: flags an LLM-based reranker needs beyond `--runner pooling`.
# Kept in the image (not in pod env) because they are JSON + a template file.
PRESET=()
case "${RERANK_MODEL_ID}" in
  Qwen/Qwen3-Reranker-*)
    # The original Qwen3-Reranker checkpoints score via the "no"/"yes" token
    # logits; these overrides turn that into a sequence-classification head, and
    # the chat template wraps each (query, document) pair the way the model was
    # trained. Recipe: vLLM v0.25.1 examples/pooling/score/qwen3_reranker_online.py.
    PRESET=(
      --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
      --chat-template /opt/podlink/templates/qwen3_reranker.jinja
    )
    ;;
esac

# Free-form extra flags for models without a preset (e.g. zerank-2). Word-split
# like VLLM_EXTRA_ARGS in start-vllm.sh — values must not contain spaces.
EXTRA_ARGS=()
if [[ -n "${RERANK_VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${RERANK_VLLM_EXTRA_ARGS}"
fi

# Never let a key reach vLLM's argv: vLLM logs every non-default CLI arg at INFO,
# unredacted (v0.25.1 log_non_default_args), and argv shows in `ps`. The key comes
# from VLLM_API_KEY in the env only.
# (vLLM's parser accepts --api_key too, and --flag=value forms.)
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    --api-key|--api-key=*|--api_key|--api_key=*) api_key_on_argv=1 ;;
  esac
done
if [[ -n "${api_key_on_argv:-}" ]]; then
  echo "[start-reranker] ERROR: --api-key in the extra args would be logged in plain text; set VLLM_API_KEY instead" >&2
  exit 1
fi

# Compile caches on the persistent volume (see vllm-cache-env.sh).
source "$(dirname "$0")/vllm-cache-env.sh"
podlink_vllm_cache_env start-reranker 18081

# Small memory share: a 4B reranker is ~8 GB bf16 weights + a little KV. 4096
# tokens covers query + one ~512-token chunk + the template with room to spare.
exec vllm serve "${RERANK_MODEL_ID}" \
  --runner pooling \
  --served-model-name "${RERANK_MODEL_ID}" \
  --gpu-memory-utilization "${RERANK_GPU_MEMORY_UTILIZATION:-0.12}" \
  --max-model-len "${RERANK_MAX_MODEL_LEN:-4096}" \
  --host 127.0.0.1 \
  --port 18081 \
  "${PRESET[@]}" \
  "${EXTRA_ARGS[@]}"
