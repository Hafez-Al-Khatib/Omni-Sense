#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Omni-Sense GCP Teardown Script
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
DB_INSTANCE_NAME="${DB_INSTANCE_NAME:-omni-sense-db}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: GCP_PROJECT_ID not set."
  exit 1
fi

echo "🧨 Tearing down Omni-Sense from project: ${PROJECT_ID}"

for svc in eep iep2 iep3 iep4; do
  echo "  Deleting omni-sense-${svc} ..."
  gcloud run services delete "omni-sense-${svc}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet || true
done

if gcloud compute instances describe "${DB_INSTANCE_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  Deleting GCE instance ${DB_INSTANCE_NAME} ..."
  gcloud compute instances delete "${DB_INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --quiet || true
fi

if gcloud compute firewall-rules describe allow-cloudrun-to-db --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  Deleting firewall rule allow-cloudrun-to-db ..."
  gcloud compute firewall-rules delete allow-cloudrun-to-db \
    --project="${PROJECT_ID}" \
    --quiet || true
fi

echo "✅ Teardown complete."
