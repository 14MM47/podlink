# Sourced by start.sh when the active provider is gcp — NOT executable on its
# own. Runs before the venv exists. Nothing interactive is needed here (auth is
# Application Default Credentials, set up once with gcloud); this only says
# early what the Python preflight will say precisely.

if ! command -v gcloud >/dev/null 2>&1; then
  warn "gcloud is not on PATH — needed for 'gcloud auth application-default login' and IAP tunnels."
fi
if [[ -z "${PODLINK_GCP_PROJECT:-}" ]]; then
  warn "PODLINK_GCP_PROJECT is not set — the preflight will fail until it is."
else
  say "Project: ${PODLINK_GCP_PROJECT} · zones: ${PODLINK_GCP_ZONES:-europe-west2-b,europe-west2-c}"
fi
