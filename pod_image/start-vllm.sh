#!/usr/bin/env bash
# Launch vLLM (the LLM service) on :8000, OpenAI-compatible /v1.
#
# The vLLM API key is read by vLLM natively from $VLLM_API_KEY — we deliberately
# do NOT pass --api-key on argv, so the token never shows up in `ps aux` inside
# the pod (the RunPod web terminal cannot be disabled).
set -euo pipefail

# --served-model-name comes from the pod env (podlink passes LLM_SERVED_NAME, from
# PODLINK_LLM_SERVED_NAME). A client's model field must match it exactly or requests
# 404. Defaults to "llm".
SERVED_NAME="${LLM_SERVED_NAME:-llm}"

# Quantization is optional: an already-quantized checkpoint (e.g. an official FP8
# build) is auto-detected, so only pass --quantization when LLM_QUANT is set
# (e.g. "awq_marlin" for a W4A16 AWQ build). Never use nvfp4 on sm_120 — the
# native MoE NVFP4 path is broken on this card.
QUANT_FLAG=()
if [[ -n "${LLM_QUANT:-}" ]]; then
  QUANT_FLAG=(--quantization "${LLM_QUANT}")
fi

# Optional extra vLLM flags, space-separated (e.g. "--language-model-only" to
# serve a multimodal checkpoint text-only, shedding vision weights + the
# encoder cache). Deliberately word-split.
EXTRA_ARGS=()
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${VLLM_EXTRA_ARGS}"
fi

exec vllm serve "${LLM_MODEL_ID}" \
  --served-model-name "${SERVED_NAME}" \
  "${QUANT_FLAG[@]}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --host 0.0.0.0 \
  --port 8000 \
  "${EXTRA_ARGS[@]}"
