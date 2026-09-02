#!/usr/bin/env bash
# One-time project setup for the podlink GCP provider. Idempotent: re-run freely.
#
# The project starts STANDALONE (no organization) and is migrated into an
# Assured Workloads folder later — see MIGRATION.md. Until then there is no org
# policy enforcing anything, so this script builds the project the way the
# folder will expect to find it, and the provider refuses to launch otherwise:
#
#   * every resource in ONE region (default europe-west2)
#   * a regional log bucket, with the _Default sink routed to it (entries
#     cannot be moved later — do this before the first VM boots)
#   * a Cloud KMS key in-region; disks and the registry are CMEK from day one
#     (a disk's key cannot be changed afterwards)
#   * a dedicated, minimally-scoped VM service account — never the default
#     compute SA
#   * the two stack secrets in Secret Manager, replicated only in-region, with
#     accessor granted per-secret to that account
#   * an Artifact Registry repo in-region for the container image
#   * firewall: IAP range only, on the stack's ports, to the podlink tag
#   * the Hyperdisk Balanced data disk (CMEK) the VM mounts at /workspace
#   * optionally a budget with alerts, and Essential Contacts
#
# Requires: gcloud authenticated as the project owner (the account that will
# later do the migration), and ~/.config/podlink/{pod_bearer_token,hf_token}.
#
#   PODLINK_GCP_PROJECT=ragline-uk-spike ./deploy/gcp/setup.sh
#   ./deploy/gcp/setup.sh --mirror ghcr.io/you/ragline-pod:2026-07b   # also mirror the image
#   PODLINK_GCP_BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX PODLINK_GCP_BUDGET_GBP=400 ./deploy/gcp/setup.sh
set -euo pipefail

PROJECT="${PODLINK_GCP_PROJECT:?set PODLINK_GCP_PROJECT}"
REGION="${PODLINK_GCP_REGION:-europe-west2}"
ZONE="${PODLINK_GCP_ZONE:-${REGION}-b}"
SA_NAME="${PODLINK_GCP_SA_NAME:-podlink-vm}"
KEYRING="${PODLINK_GCP_KEYRING:-podlink}"
KEY="${PODLINK_GCP_KEY:-disks}"
REPO="${PODLINK_GCP_REPO:-podlink}"
DISK="${PODLINK_GCP_DATA_DISK:-podlink-data}"
DISK_GB="${PODLINK_GCP_DATA_DISK_GB:-200}"
SECRET_PREFIX="${PODLINK_GCP_SECRET_PREFIX:-podlink}"
NETWORK="${PODLINK_GCP_NETWORK:-default}"
TAG="${PODLINK_GCP_NETWORK_TAG:-podlink}"
LOG_BUCKET="${PODLINK_GCP_LOG_BUCKET:-podlink-${REGION}}"
CONFIG_DIR="${HOME}/.config/podlink"
MIRROR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mirror) [[ $# -ge 2 ]] || { echo "--mirror needs an image ref" >&2; exit 2; }; MIRROR="$2"; shift 2 ;;
    *) echo "unknown argument: $1 (supported: --mirror <image>)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[36m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m   ✓ %s\033[0m\n' "$*"; }
note() { printf '\033[33m   ! %s\033[0m\n' "$*"; }
have() { "$@" >/dev/null 2>&1; }

gcloud config set project "${PROJECT}" >/dev/null
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
COMPUTE_AGENT="service-${PROJECT_NUMBER}@compute-system.iam.gserviceaccount.com"
AR_AGENT="service-${PROJECT_NUMBER}@gcp-sa-artifactregistry.iam.gserviceaccount.com"
KEY_RES="projects/${PROJECT}/locations/${REGION}/keyRings/${KEYRING}/cryptoKeys/${KEY}"

say "1. APIs"
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  cloudkms.googleapis.com iap.googleapis.com logging.googleapis.com monitoring.googleapis.com \
  cloudresourcemanager.googleapis.com cloudbilling.googleapis.com essentialcontacts.googleapis.com >/dev/null
ok "enabled"

say "2. Logs stay in ${REGION} (a standalone project's _Default bucket is global)"
if ! have gcloud logging buckets describe "${LOG_BUCKET}" --location="${REGION}"; then
  gcloud logging buckets create "${LOG_BUCKET}" --location="${REGION}" \
    --description="podlink: regional log storage (residency)" --retention-days=30 >/dev/null
fi
gcloud logging sinks update _Default \
  "logging.googleapis.com/projects/${PROJECT}/locations/${REGION}/buckets/${LOG_BUCKET}" >/dev/null
ok "_Default sink -> ${LOG_BUCKET} (${REGION})"
note "_Required (admin-activity audit) stays global on a standalone project — see MIGRATION.md"

say "3. Cloud KMS key in ${REGION} (CMEK for disks + registry)"
have gcloud kms keyrings describe "${KEYRING}" --location="${REGION}" \
  || gcloud kms keyrings create "${KEYRING}" --location="${REGION}"
have gcloud kms keys describe "${KEY}" --keyring="${KEYRING}" --location="${REGION}" \
  || gcloud kms keys create "${KEY}" --keyring="${KEYRING}" --location="${REGION}" --purpose=encryption \
       --rotation-period=90d --next-rotation-time="$(date -u -d '+90 days' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v+90d +%Y-%m-%dT%H:%M:%SZ)"
# The Compute Engine and Artifact Registry service agents encrypt on the project's behalf.
for agent in "${COMPUTE_AGENT}" "${AR_AGENT}"; do
  gcloud kms keys add-iam-policy-binding "${KEY}" --keyring="${KEYRING}" --location="${REGION}" \
    --member="serviceAccount:${agent}" --role=roles/cloudkms.cryptoKeyEncrypterDecrypter >/dev/null 2>&1 || true
done
ok "${KEY_RES} (90-day rotation)"

say "4. Dedicated VM identity ${SA} (minimal roles)"
have gcloud iam service-accounts describe "${SA}" \
  || gcloud iam service-accounts create "${SA_NAME}" --display-name="podlink GPU VM" >/dev/null
for role in roles/logging.logWriter roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT}" --member="serviceAccount:${SA}" --role="${role}" \
    --condition=None >/dev/null
done
ok "logWriter + metricWriter (registry + secrets are granted per-resource below)"

say "5. Secrets in Secret Manager, replicated only in ${REGION}"
for pair in "${SECRET_PREFIX}-bearer:pod_bearer_token" "${SECRET_PREFIX}-hf-token:hf_token"; do
  name="${pair%%:*}"; file="${CONFIG_DIR}/${pair##*:}"
  [[ -f "${file}" ]] || { echo "missing ${file}" >&2; exit 1; }
  if ! have gcloud secrets describe "${name}"; then
    gcloud secrets create "${name}" --replication-policy=user-managed --locations="${REGION}" >/dev/null
  fi
  gcloud secrets versions add "${name}" --data-file="${file}" >/dev/null
  gcloud secrets add-iam-policy-binding "${name}" --member="serviceAccount:${SA}" \
    --role=roles/secretmanager.secretAccessor >/dev/null
  ok "${name}: new version from ${file##*/}; accessor -> ${SA_NAME}"
done

say "6. Artifact Registry ${REGION}-docker.pkg.dev/${PROJECT}/${REPO} (CMEK)"
have gcloud artifacts repositories describe "${REPO}" --location="${REGION}" \
  || gcloud artifacts repositories create "${REPO}" --repository-format=docker --location="${REGION}" \
       --kms-key="${KEY_RES}" --description="podlink container images (in-region mirror)" >/dev/null
gcloud artifacts repositories add-iam-policy-binding "${REPO}" --location="${REGION}" \
  --member="serviceAccount:${SA}" --role=roles/artifactregistry.reader >/dev/null
ok "reader -> ${SA_NAME}"
if [[ -n "${MIRROR}" ]]; then
  command -v docker >/dev/null || { echo "docker is required for --mirror" >&2; exit 1; }
  target="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${MIRROR##*/}"
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet >/dev/null
  docker pull "${MIRROR}" && docker tag "${MIRROR}" "${target}" && docker push "${target}"
  ok "mirrored -> ${target}   (set PODLINK_GCP_IMAGE to this)"
fi

say "7. Firewall on network '${NETWORK}': IAP range only, to tag '${TAG}'"
have gcloud compute firewall-rules describe podlink-allow-iap \
  || gcloud compute firewall-rules create podlink-allow-iap --network="${NETWORK}" --direction=INGRESS \
       --priority=900 --action=ALLOW --rules=tcp:22,tcp:8000,tcp:8080,tcp:8081 \
       --source-ranges=35.235.240.0/20 --target-tags="${TAG}" >/dev/null
have gcloud compute firewall-rules describe podlink-deny-ingress \
  || gcloud compute firewall-rules create podlink-deny-ingress --network="${NETWORK}" --direction=INGRESS \
       --priority=1000 --action=DENY --rules=all --source-ranges=0.0.0.0/0 --target-tags="${TAG}" >/dev/null
ok "allow 35.235.240.0/20 -> 22,8000,8080,8081; deny everything else to the tag"

say "8. Data disk ${DISK}: ${DISK_GB} GB Hyperdisk Balanced in ${ZONE}, CMEK"
have gcloud compute disks describe "${DISK}" --zone="${ZONE}" \
  || gcloud compute disks create "${DISK}" --zone="${ZONE}" --type=hyperdisk-balanced --size="${DISK_GB}GB" \
       --kms-key="${KEY_RES}" --labels=managed-by=podlink >/dev/null
ok "attach as deviceName podlink-data; the VM formats it on first boot"

if [[ -n "${PODLINK_GCP_BILLING_ACCOUNT:-}" && -n "${PODLINK_GCP_BUDGET_GBP:-}" ]]; then
  say "9. Budget £${PODLINK_GCP_BUDGET_GBP}/month with alerts at 50/90/100%"
  gcloud billing budgets create --billing-account="${PODLINK_GCP_BILLING_ACCOUNT}" \
    --display-name="podlink ${PROJECT}" --budget-amount="${PODLINK_GCP_BUDGET_GBP}GBP" \
    --filter-projects="projects/${PROJECT_NUMBER}" \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 >/dev/null \
    && ok "created (storage SKUs included — disks bill without a VM)" || note "budget create failed (exists? permissions?)"
fi
if [[ -n "${PODLINK_GCP_CONTACT_EMAIL:-}" ]]; then
  say "10. Essential Contacts -> ${PODLINK_GCP_CONTACT_EMAIL}"
  gcloud essential-contacts create --email="${PODLINK_GCP_CONTACT_EMAIL}" \
    --notification-categories=billing,security,technical,suspension --project="${PROJECT}" >/dev/null 2>&1 \
    && ok "billing, security, technical, suspension" || note "contact already present or not permitted"
fi

say "Done. Profile block for ~/.config/podlink/profiles/<name>.conf:"
cat <<EOF

export PODLINK_PROVIDER=gcp
export PODLINK_GCP_PROJECT=${PROJECT}
export PODLINK_GCP_ZONES=${ZONE},${REGION}-c
export PODLINK_GCP_BOOT_IMAGE=projects/${PROJECT}/global/images/family/podlink-g4   # the golden image (Phase 2)
export PODLINK_GCP_IMAGE=${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${MIRROR:+${MIRROR##*/}}${MIRROR:-<image:tag>}
export PODLINK_GCP_DATA_DISK=${DISK}
export PODLINK_GCP_KMS_KEY=${KEY_RES}
export PODLINK_GCP_SERVICE_ACCOUNT=${SA}
export PODLINK_GCP_NETWORK_TAG=${TAG}
export PODLINK_GCP_COST_PER_HR=5.40           # + the NVIDIA CC licence once priced
export PODLINK_GCP_HARDENING=strict           # the default; relaxed is for non-sensitive experiments only
EOF
