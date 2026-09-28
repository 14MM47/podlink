# Sourced (not executed) by the three service wrappers: GPU pinning for multi-GPU
# pods, compute-capability detection, and the choice of TEI router binary.
#
# One image serves more than one card. vLLM's base carries kernels for every
# supported arch, but TEI's router is built per compute capability, so the image
# bundles one binary per supported arch (/usr/local/bin/text-embeddings-router-<arch>)
# and the TEI wrappers pick at boot from what nvidia-smi reports:
#   12.x           -> sm120  (RTX PRO 6000 Blackwell, RTX 50x0)
#   8.6 / 8.7 / 8.9 -> sm86  (RTX 3090 / A10 / A40; Ada runs the sm_86 build)
# TEI_ARCH=<arch> forces one; PODLINK_COMPUTE_CAP=<x.y> overrides detection (tests,
# or a host without nvidia-smi). No bundled build for the detected card is an
# error — better a red tile than a router that dies on its first kernel.

# podlink_pin_gpu <who> <indices>: restrict this service to the given GPU
# index(es) via CUDA_VISIBLE_DEVICES. Empty = unchanged (every GPU visible), so a
# single-GPU pod behaves exactly as before.
podlink_pin_gpu() {
  local who="$1" gpus="$2"
  if [[ -n "${gpus}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpus}"
    echo "[${who}] pinned to GPU(s) ${gpus} (CUDA_VISIBLE_DEVICES)"
  fi
}

# podlink_compute_cap: print the compute capability (e.g. 12.0, 8.6) of this
# service's first visible GPU, or nothing when it cannot be determined.
podlink_compute_cap() {
  if [[ -n "${PODLINK_COMPUTE_CAP:-}" ]]; then
    echo "${PODLINK_COMPUTE_CAP}"
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so ask for the pinned index explicitly
  # (an index or a GPU UUID both work with -i). Pods are homogeneous anyway.
  # Callers run under `set -euo pipefail`: every step here must be non-fatal.
  local idx="${CUDA_VISIBLE_DEVICES:-}"
  idx="${idx%%,*}"
  local out
  out="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${idx:-0}" 2>/dev/null || true)"
  out="${out%%$'\n'*}"
  echo "${out// /}"
}

# podlink_tei_router <who>: print the TEI router binary to exec (logs go to stderr).
podlink_tei_router() {
  local who="$1" dir="${PODLINK_TEI_BIN_DIR:-/usr/local/bin}" arch="${TEI_ARCH:-}"
  if [[ -z "${arch}" ]]; then
    local cap
    cap="$(podlink_compute_cap)"
    case "${cap}" in
      12.*)          arch=sm120 ;;
      8.6|8.7|8.9)   arch=sm86 ;;
      "")            arch="" ;;      # unknown host: fall through to the PATH router
      *)
        echo "[${who}] ERROR: no bundled TEI build for compute capability ${cap} (bundled: sm120, sm86); set TEI_ARCH to force one" >&2
        return 1
        ;;
    esac
  fi
  if [[ -n "${arch}" && -x "${dir}/text-embeddings-router-${arch}" ]]; then
    echo "[${who}] TEI build: ${arch}" >&2
    echo "${dir}/text-embeddings-router-${arch}"
    return 0
  fi
  if [[ -n "${arch}" ]]; then
    echo "[${who}] WARNING: ${dir}/text-embeddings-router-${arch} not found; using text-embeddings-router from PATH" >&2
  fi
  echo text-embeddings-router
}
