# Sourced by start.sh when the active provider is runpod — NOT executable on
# its own. Runs before the venv exists, so it must stay plain bash. Inherits
# say/warn/ok, CONFIG_DIR and the loaded PODLINK_* env from the launcher.
#
# Resolves the Network Volume id. Priority: an already-exported env var wins
# (explicit override); else a saved file; else prompt. When a saved value exists
# we still OFFER a fresh paste. Sentinel: PODLINK_NETWORK_VOLUME_ID=none means
# "deliberately no volume" — it skips the saved-file/prompt fallback so a
# no-volume profile can't silently inherit another stack's volume, and runs in
# Data-Volume mode (weights re-download on every POD UP; nothing persists a
# terminate). app/providers/runpod/__init__.py:normalise_env mirrors this for
# the console's profile switch.

VOL_FILE="${CONFIG_DIR}/network_volume_id"     # the id is infra, not a secret

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
