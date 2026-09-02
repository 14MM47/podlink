#!/usr/bin/env bash
# Single-command launcher for podlink.
#
# Loads the stack config (+ optional profile), works out which provider this
# launch drives, gives that provider its chance to resolve anything interactive
# (its launch.sh — e.g. a saved storage id), ensures the venv + deps (core plus
# that provider's own), then either runs the preflight (--check) or hands off
# to the hardened ./run.sh (localhost-only bind, no arg pass-through — we
# deliberately do NOT forward args to it).
#
# This file is provider-neutral by design and tests/test_siloing.py keeps it
# that way: anything cloud-specific belongs in app/providers/<name>/.
#
#   ./start.sh                      # resolve prerequisites, then start the app
#   ./start.sh --check              # preflight only (config, venv, deps, secrets), no launch
#   ./start.sh --profile <name>     # overlay ~/.config/podlink/profiles/<name>.conf
#                                   # on the base conf (also: PODLINK_PROFILE env)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${HOME}/.config/podlink"
CONF_FILE="${CONFIG_DIR}/podlink.conf"         # optional: exports PODLINK_IMAGE etc.
PROFILE_DIR="${CONFIG_DIR}/profiles"           # optional per-stack overlays
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

# --- 0. Load the stack config (PODLINK_IMAGE, provider, served name, …) ------
# Account/stack-specific values live in a local, gitignored file, not in source.
if [[ -f "${CONF_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONF_FILE}"
  say "Loaded stack config from ${CONF_FILE}"
fi

# Profile overlay: a named conf sourced AFTER the base, so its exports win.
# Lets one podlink switch between whole stacks — and clouds — per launch.
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

# --- 1. Resolve the provider and let it do its interactive part --------------
# The default lives in ONE place (app/providers/__init__.py); read it rather
# than repeat it. The name feeds a path and an import below, so it is validated
# to a plain identifier first — a conf is user-owned, but not a place for
# surprises.
cd "${HERE}"
DEFAULT_PROVIDER="$(python3 -c 'from app.providers import DEFAULT_PROVIDER as d; print(d)')"
PROVIDER="${PODLINK_PROVIDER:-${DEFAULT_PROVIDER}}"
PROVIDER="${PROVIDER,,}"
if [[ ! "${PROVIDER}" =~ ^[a-z0-9_]+$ || ! -d "app/providers/${PROVIDER}" ]]; then
  warn "Unknown PODLINK_PROVIDER '${PROVIDER}' — no app/providers/${PROVIDER}/ package."
  say "Available: $(ls -d app/providers/*/ | xargs -n1 basename | grep -v __pycache__ | paste -sd' ' -)"
  exit 2
fi
export PODLINK_PROVIDER="${PROVIDER}"
say "Provider: ${PROVIDER}"

# A provider may need a TTY before the venv exists (a prompt for a saved id, a
# login). It gets that here, with say/warn/ok and CONFIG_DIR in scope.
if [[ -f "app/providers/${PROVIDER}/launch.sh" ]]; then
  # shellcheck disable=SC1090
  source "app/providers/${PROVIDER}/launch.sh"
fi

# --- 2. Ensure venv + deps (core, then the provider's own) -------------------
if [[ ! -d .venv ]]; then
  say "Creating .venv …"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# Install only when something is missing — keeps a warm start fast.
if ! python -c "import uvicorn, fastapi, httpx" 2>/dev/null; then
  say "Installing core dependencies …"
  pip install -q -r requirements.txt
fi
# Importing the provider package pulls in its SDK; a failure means it's not
# installed yet. Each provider's SDK is pinned in its own requirements file so
# a launch never installs another cloud's.
if ! python -c "import app.providers.${PROVIDER}" 2>/dev/null; then
  if [[ -f "requirements-${PROVIDER}.txt" ]]; then
    say "Installing ${PROVIDER} provider dependencies …"
    pip install -q -r "requirements-${PROVIDER}.txt"
  fi
  python -c "import app.providers.${PROVIDER}" || {
    warn "The ${PROVIDER} provider failed to import — see the traceback above."; exit 2; }
fi
ok "Environment ready (venv + core + ${PROVIDER} deps)."

# --- 3. Preflight report for --check ---------------------------------------
# Lives in Python (app/preflight.py) so the active provider owns its own checks.
if [[ "${CHECK_ONLY}" == "1" ]]; then
  say ""
  exec python -m app.preflight
fi

# --- 4. Launch (hardened localhost bind lives in run.sh) --------------------
say "Starting podlink on http://127.0.0.1:8765 …"
exec ./run.sh
