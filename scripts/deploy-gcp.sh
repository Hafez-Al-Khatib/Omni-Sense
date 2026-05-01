#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Omni-Sense GCP Cloud Run Deploy Script
# Fallback for users without Terraform installed.
# Requires: gcloud CLI authenticated + project set
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
REGISTRY="${IMAGE_REGISTRY:-docker.io}"
TAG="${IMAGE_TAG:-latest}"
DB_INSTANCE_NAME="${DB_INSTANCE_NAME:-omni-sense-db}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: GCP_PROJECT_ID not set and no default project configured."
  echo "Run: gcloud auth login && gcloud config set project <PROJECT_ID>"
  exit 1
fi

echo "🚀 Deploying Omni-Sense to GCP project: ${PROJECT_ID} region: ${REGION}"

# ── Enable APIs ───────────────────────────────────────────────────────────────
echo "📡 Enabling APIs..."
gcloud services enable run.googleapis.com compute.googleapis.com --project="${PROJECT_ID}"

# ── Deploy Cloud Run services ─────────────────────────────────────────────────
deploy_service() {
  local name=$1
  local image=$2
  local port=$3
  local min_inst=$4
  local max_inst=$5
  local cpu=$6
  local memory=$7

  echo "⛵ Deploying ${name} ..."
  gcloud run deploy "omni-sense-${name}" \
    --image="${image}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --port="${port}" \
    --min-instances="${min_inst}" \
    --max-instances="${max_inst}" \
    --cpu="${cpu}" \
    --memory="${memory}" \
    --timeout=300 \
    --concurrency=80 \
    --allow-unauthenticated \
    --platform managed \
    --quiet || true
}

# Deploy IEP2 first (so EEP can reference its URL later)
deploy_service "iep2" "${REGISTRY}/omni-sense-iep2:${TAG}" 8002 0 5 1 512Mi
deploy_service "iep3" "${REGISTRY}/omni-sense-iep3:${TAG}" 8003 0 5 1 512Mi
deploy_service "iep4" "${REGISTRY}/omni-sense-iep4:${TAG}" 8004 1 5 2 2Gi

IEP2_URL=$(gcloud run services describe omni-sense-iep2 --region="${REGION}" --format='value(status.url)' --project="${PROJECT_ID}")
IEP3_URL=$(gcloud run services describe omni-sense-iep3 --region="${REGION}" --format='value(status.url)' --project="${PROJECT_ID}")
IEP4_URL=$(gcloud run services describe omni-sense-iep4 --region="${REGION}" --format='value(status.url)' --project="${PROJECT_ID}")

deploy_service "eep" "${REGISTRY}/omni-sense-eep:${TAG}" 8000 0 5 1 512Mi

# Update EEP env vars to point to sibling services
echo "🔗 Linking EEP to downstream services..."
gcloud run services update omni-sense-eep \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --update-env-vars="OMNI_IEP2_URL=${IEP2_URL},OMNI_IEP3_URL=${IEP3_URL},OMNI_IEP4_URL=${IEP4_URL},OMNI_RATE_LIMIT=5/minute" \
  --quiet || true

# ── GCE instance for TimescaleDB + Redis ──────────────────────────────────────
if [[ "${ENABLE_DB_INSTANCE:-true}" == "true" ]]; then
  echo "🗄️  Creating DB instance ${DB_INSTANCE_NAME} (if not exists)..."
  if ! gcloud compute instances describe "${DB_INSTANCE_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud compute instances create "${DB_INSTANCE_NAME}" \
      --zone="${ZONE}" \
      --project="${PROJECT_ID}" \
      --machine-type=e2-small \
      --image-family=ubuntu-2204-lts \
      --image-project=ubuntu-os-cloud \
      --boot-disk-size=20GB \
      --tags=omni-sense-db \
      --metadata-from-file startup-script="${SCRIPT_DIR}/db-startup.sh" \
      --quiet
  else
    echo "   Instance ${DB_INSTANCE_NAME} already exists, skipping creation."
  fi

  echo "🔓 Opening firewall for DB ports..."
  if ! gcloud compute firewall-rules describe allow-cloudrun-to-db --project="${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud compute firewall-rules create allow-cloudrun-to-db \
      --allow tcp:5432,tcp:6379 \
      --source-ranges=0.0.0.0/0 \
      --target-tags=omni-sense-db \
      --project="${PROJECT_ID}" \
      --quiet
  fi
fi

# ── Print URLs ────────────────────────────────────────────────────────────────
echo ""
echo "✅ Deployment complete! Service URLs:"
echo "  EEP:  $(gcloud run services describe omni-sense-eep   --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID})"
echo "  IEP2: $(gcloud run services describe omni-sense-iep2 --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID})"
echo "  IEP3: $(gcloud run services describe omni-sense-iep3 --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID})"
echo "  IEP4: $(gcloud run services describe omni-sense-iep4 --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID})"
