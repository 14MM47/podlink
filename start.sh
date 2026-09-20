#!/usr/bin/env bash
# Single-command launcher for podlink.
#
# Resolves the RunPod Network Volume id (env -> saved file -> interactive prompt,
# always offering you the chance to paste a new one), ensures the venv + deps are
# ready, then hands off to the hardened ./run.sh (localhost-only bind, no arg
# pass-through — we deliberately do NOT forward args to it).
#
#   ./start.sh                      # resolve volume id, then start the app
#   ./start.sh --check              # preflight only (volume, venv, deps, secrets), no launch
#   ./start.sh --profile <name>     # overlay ~/.config/podlink/profiles/<name>.conf
#                                   # on the base conf (also: PODLINK_PROFILE env)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${HOME}/.config/podlink"
CONF_FILE="${CONFIG_DIR}/podlink.conf"         # optional: exports PODLINK_IMAGE etc.
PROFILE_DIR="${CONFIG_DIR}/profiles"           # optional per-stack overlays
VOL_FILE="${CONFIG_DIR}/network_volume_id"     # the id is infra, not a secret, but kept here for tidiness
CHECK_ONLY=0
PROFILE="${PODLINK_PROFILE:-}"                 # --profile beats the env var
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)   CHECK_ONLY=1; shift ;;
    --profile) [[ $# -ge 2 ]] || { printf '\033[33m%s\033[0m\n' "--profile needs a name"; exit 2; }
               PROFILE="$2"; shift 2 ;;
    *)         printf '\033[33m%s\033[0m\n' "Unknown argument: $1 (supported: --check, --profile <name>)"; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m%s\033[0m\n' "$*"; }

# --- 0. Load the stack config (PODLINK_IMAGE, registry auth, served name, …) ---
# Account/stack-specific values live in a local, gitignored file, not in source.
if [[ -f "${CONF_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONF_FILE}"
  say "Loaded stack config from ${CONF_FILE}"
fi

# Profile overlay: a named conf sourced AFTER the base, so its exports win.
# Lets one podlink switch between whole stacks (models, sizing, volume) per launch.
if [[ -n "${PROFILE}" ]]; then
  PROFILE_FILE="${PROFILE_DIR}/${PROFILE}.conf"
  if [[ ! -f "${PROFILE_FILE}" ]]; then
    warn "Profile '${PROFILE}' not found at ${PROFILE_FILE}"
    [[ -d "${PROFILE_DIR}" ]] && say "Available: $(ls "${PROFILE_DIR}" 2>/dev/null | sed 's/\.conf$//' | paste -sd' ' -)"
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${PROFILE_FILE}"
  export PODLINK_PROFILE="${PROFILE}"    # surfaced in the console UI
  say "Loaded profile '${PROFILE}' from ${PROFILE_FILE}"
fi

# --- 1. Resolve the Network Volume id --------------------------------------
# Priority: an already-exported env var wins (explicit override); else a saved
# file; else prompt. When a saved value exists we still OFFER a fresh paste.
# Sentinel: PODLINK_NETWORK_VOLUME_ID=none means "deliberately no volume" — it
# skips the saved-file/prompt fallback so a no-volume profile can't silently
# inherit another stack's volume, and runs in Data-Volume mode (weights
# re-download from HF on every POD UP; nothing persists a terminate).
volume=""
if [[ "${PODLINK_NETWORK_VOLUME_ID:-}" =~ ^[Nn][Oo][Nn][Ee]$ ]]; then
  say "Network volume: none (explicit) — Data-Volume mode, weights re-download each POD UP."
elif [[ -n "${PODLINK_NETWORK_VOLUME_ID:-}" ]]; then
  volume="${PODLINK_NETWORK_VOLUME_ID}"
  say "Network volume id: using PODLINK_NETWORK_VOLUME_ID from the environment."
elif [[ -f "${VOL_FILE}" ]]; then
  saved="$(<"${VOL_FILE}")"; saved="${saved//[[:space:]]/}"
  if [[ -t 0 ]]; then
    read -rp "Network volume id [${saved}] (Enter to keep, or paste a new one): " entered || entered=""
    entered="${entered//[[:space:]]/}"
    if [[ -n "${entered}" && "${entered}" != "${saved}" ]]; then
      volume="${entered}"
      printf '%s' "${volume}" > "${VOL_FILE}"; chmod 600 "${VOL_FILE}"
      ok "Updated saved id in ${VOL_FILE}."
    else
      volume="${saved}"
    fi
  else
    volume="${saved}"          # non-interactive: fall back to the saved value
  fi
  say "Network volume id: ${volume:-<none>}"
else
  if [[ -t 0 ]]; then
    read -rp "Paste your RunPod Network Volume id (blank = no-volume / Data-Volume mode): " entered || entered=""
    volume="${entered//[[:space:]]/}"
    if [[ -n "${volume}" ]]; then
      read -rp "Save it to ${VOL_FILE} for next time? [Y/n] " ans || ans=""
      if [[ ! "${ans}" =~ ^[Nn] ]]; then
        mkdir -p "${CONFIG_DIR}"; chmod 700 "${CONFIG_DIR}"
        printf '%s' "${volume}" > "${VOL_FILE}"; chmod 600 "${VOL_FILE}"
        ok "Saved."
      fi
    fi
  else
    warn "No network volume id set and no TTY to prompt on."
  fi
fi

# Soft format sanity — RunPod ids are lowercase alphanumeric. Warn, don't block,
# since the exact format isn't contractual.
if [[ -n "${volume}" && ! "${volume}" =~ ^[a-z0-9]{6,}$ ]]; then
  warn "Note: '${volume}' doesn't look like a typical RunPod volume id — double-check it."
fi
if [[ -z "${volume}" ]]; then
  warn "No Network Volume set — POD DOWN will DESTROY downloaded weights (Data-Volume fallback)."
fi
export PODLINK_NETWORK_VOLUME_ID="${volume}"

# --- 2. Ensure venv + deps --------------------------------------------------
cd "${HERE}"
if [[ ! -d .venv ]]; then
  say "Creating .venv …"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# Install only when something is missing — keeps a warm start fast.
if ! python -c "import uvicorn, fastapi, runpod, httpx, rich" 2>/dev/null; then
  say "Installing dependencies …"
  pip install -q -r requirements.txt
fi
ok "Environment ready (venv + deps)."

# --- 3. Preflight report for --check ---------------------------------------
if [[ "${CHECK_ONLY}" == "1" ]]; then
  say ""
  say "Secrets in ${CONFIG_DIR}:"
  for f in runpod_api_key hf_token pod_bearer_token; do
    if [[ -f "${CONFIG_DIR}/${f}" ]]; then
      mode="$(stat -c '%a' "${CONFIG_DIR}/${f}" 2>/dev/null || echo '?')"
      if [[ "${mode}" == "600" ]]; then ok "  ✓ ${f} (0600)"; else warn "  ! ${f} (mode ${mode}, want 600)"; fi
    else
      warn "  ✗ ${f} MISSING"
    fi
  done
  say ""
  say "Stack config:"
  say "  profile: ${PODLINK_PROFILE:-<none — base conf only>}"
  if [[ -n "${PODLINK_IMAGE:-}" ]]; then ok "  ✓ PODLINK_IMAGE = ${PODLINK_IMAGE}"; else warn "  ✗ PODLINK_IMAGE not set (required — see README / ${CONF_FILE})"; fi
  say "  served model name: ${PODLINK_LLM_SERVED_NAME:-llm}"
  say "  llm:      ${PODLINK_LLM_MODEL_ID:-<pod_up.py default>}"
  say "  embedder: ${PODLINK_EMBED_MODEL_ID:-<pod_up.py default>}"
  say "  reranker: ${PODLINK_RERANK_MODEL_ID:-<pod_up.py default>}"
  say "  max model len: ${PODLINK_MAX_MODEL_LEN:-32768} · gpu share: ${PODLINK_GPU_MEMORY_UTILIZATION:-0.70}"
  say "  gpu: ${PODLINK_GPU_MATCH:-RTX PRO 6000} (>= ${PODLINK_GPU_MIN_VRAM_GB:-90} GB) · container disk: ${PODLINK_CONTAINER_DISK_GB:-55} GB"
  # Validate the service spec with the same parser the app uses, so a profile typo
  # fails here instead of deploying a pod whose tiles never go green.
  if svc="$(python - <<'PY'
import sys; sys.path.insert(0, "pod_control")
import pod_services
try:
    print(", ".join(f"{n}:{p}" for n, (p, _h) in pod_services.parse_services().items()))
except ValueError as e:
    print(e, file=sys.stderr); sys.exit(1)
PY
)"; then
    say "  services: ${svc}"
  else
    warn "  ✗ PODLINK_SERVICES is malformed (see error above)"
  fi
  extra="$(env | grep -c '^PODLINK_POD_ENV_' || true)"
  if [[ "${extra}" != "0" ]]; then say "  pod env passthrough: ${extra} PODLINK_POD_ENV_* value(s)"; fi
  if [[ -n "${PODLINK_REGISTRY_AUTH_ID:-}" ]]; then say "  registry auth: set (private image)"; else say "  registry auth: none (public image)"; fi
  say ""
  if [[ -n "${PODLINK_NETWORK_VOLUME_ID}" ]]; then
    say "Volume mode: network volume ${PODLINK_NETWORK_VOLUME_ID}"
  else
    warn "Volume mode: Data-Volume fallback (no persistence)"
  fi
  ok "Preflight OK — run ./start.sh (no --check) to launch."
  exit 0
fi

# --- 4. Launch (hardened localhost bind lives in run.sh) --------------------
say "Starting podlink on http://127.0.0.1:8765 …"
exec ./run.sh
