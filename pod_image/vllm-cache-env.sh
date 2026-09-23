# Sourced (not executed) by start-vllm.sh and start-reranker.sh just before
# `exec vllm serve`: put vLLM's compile caches on the persistent volume.
#
# vLLM keeps torch.compile artifacts under VLLM_CACHE_ROOT (default ~/.cache/vllm;
# FlashInfer's autotune cache lives there too — logged 2026-09-23 as
# /root/.cache/vllm/flashinfer_autotune_cache/…) and Triton keeps kernels under
# ~/.triton. Both are on the container disk, which a stop or a fresh pod wipes,
# so every boot recompiled (~14 s reranker + ~17 s LLM, measured 2026-09-23). On
# the volume they survive, and later boots load the compiled graphs instead.
#
# Layout: <base>/<service>/{vllm,triton}-<vllm version>/ — one tree per service
# (the LLM and the vLLM reranker never share a directory, so they can't race on
# the same kernel file over the FUSE volume) and per vLLM version (an upgraded
# image never loads an incompatible cache). Explicit VLLM_CACHE_ROOT /
# TRITON_CACHE_DIR always win; if the base isn't writable (no volume mounted)
# the container-disk defaults are kept unchanged. PODLINK_CACHE_BASE overrides
# the base (tests).
#
# Self-heal: a cache on the volume outlives a crashed start, so a half-written
# artifact could fail every later boot. Each start drops a marker that a small
# background watcher removes once the service answers /health. A start that
# finds the marker still present knows the previous start never became healthy,
# and deletes this service's cache so it is rebuilt — at most one lost start.
# It's a cache: deleting it is always safe.

podlink_vllm_cache_env() {
  local who="$1"                                   # service name, e.g. start-vllm
  local port="$2"                                  # its loopback port, for /health
  local base="${PODLINK_CACHE_BASE:-/workspace/cache}"
  local ver
  # Package metadata, not `import vllm` (that import takes ~10 s).
  ver="$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || true)"
  [[ -n "${ver}" ]] || ver="unknown"
  local root="${base}/${who}"
  if ! { mkdir -p "${root}" 2>/dev/null && [[ -w "${root}" ]]; }; then
    echo "[${who}] ${base} not writable; keeping the container-disk compile caches" >&2
    return 0
  fi

  local vdir="${root}/vllm-${ver}" tdir="${root}/triton-${ver}"
  local marker="${root}/.start-in-progress"
  if [[ -e "${marker}" ]]; then
    # Previous start never reached healthy: its cache may be half-written.
    echo "[${who}] previous start never became healthy; clearing its compile cache" >&2
    rm -rf -- "${vdir}" "${tdir}"
  fi
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${vdir}}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${tdir}}"
  echo "[${who}] compile caches: VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT} TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

  # Mark this start, then clear the mark from a detached watcher once /health
  # answers (it outlives the `exec` below). Bounded: PODLINK_CACHE_HEALTH_TRIES
  # probes 10 s apart (default 360 = 1 h), so it can never linger forever.
  touch "${marker}"
  local tries="${PODLINK_CACHE_HEALTH_TRIES:-360}"
  [[ "${tries}" =~ ^[0-9]+$ ]] || tries=360
  (
    for ((i = 0; i < tries; i++)); do
      if curl -fsS -m 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        rm -f -- "${marker}"
        exit 0
      fi
      sleep 10
    done
  ) >/dev/null 2>&1 &
}
