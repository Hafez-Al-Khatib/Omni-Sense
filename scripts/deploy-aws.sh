#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Omni-Sense AWS App Runner Deploy Script
# Alternative to GCP for students who learned AWS in class.
# Requires: aws CLI authenticated + default region set
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
TAG="${IMAGE_TAG:-latest}"

echo "🚀 Deploying Omni-Sense to AWS region: ${REGION}"

# ── Login to ECR ──────────────────────────────────────────────────────────────
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"

# ── Create ECR repos (idempotent) ─────────────────────────────────────────────
for repo in omni-sense-eep omni-sense-iep2 omni-sense-iep3 omni-sense-iep4; do
  aws ecr describe-repositories --repository-names "${repo}" --region "${REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${repo}" --region "${REGION}"
done

# ── Build & Push images ───────────────────────────────────────────────────────
echo "🐳 Building images..."
docker build -t "${REGISTRY}/omni-sense-eep:${TAG}" ./eep
docker build -t "${REGISTRY}/omni-sense-iep2:${TAG}" -f iep2/Dockerfile .
docker build -t "${REGISTRY}/omni-sense-iep3:${TAG}" -f iep3/Dockerfile .
docker build -t "${REGISTRY}/omni-sense-iep4:${TAG}" -f iep4/Dockerfile .

echo "📤 Pushing images..."
for svc in eep iep2 iep3 iep4; do
  docker push "${REGISTRY}/omni-sense-${svc}:${TAG}"
done

# ── Deploy to App Runner ──────────────────────────────────────────────────────
# App Runner is AWS's serverless container platform (like GCP Cloud Run)
deploy_apprunner() {
  local name=$1
  local image=$2
  local port=$3

  echo "⛵ Deploying ${name} to App Runner..."
  
  # Check if service exists
  if aws apprunner list-services --region "${REGION}" | grep -q "\"ServiceName\": \"${name}\""; then
    # Update existing service
    aws apprunner update-service \
      --service-arn "$(aws apprunner list-services --region ${REGION} --query 'ServiceSummaryList[?ServiceName==\`${name}\`].ServiceArn' --output text)" \
      --source-configuration "ImageRepository={ImageIdentifier=${image},ImageRepositoryType=ECR},AutoDeploymentsEnabled=false,ImageConfiguration={Port=${port}}}" \
      --region "${REGION}" \
      --no-cli-pager
  else
    # Create new service
    aws apprunner create-service \
      --service-name "${name}" \
      --source-configuration "ImageRepository={ImageIdentifier=${image},ImageRepositoryType=ECR},AutoDeploymentsEnabled=false,ImageConfiguration={Port=${port}}}" \
      --instance-configuration "Cpu=1 vCPU,Memory=2 GB" \
      --region "${REGION}" \
      --no-cli-pager
  fi
}

deploy_apprunner "omni-sense-iep2" "${REGISTRY}/omni-sense-iep2:${TAG}" 8002
deploy_apprunner "omni-sense-iep3" "${REGISTRY}/omni-sense-iep3:${TAG}" 8003
deploy_apprunner "omni-sense-iep4" "${REGISTRY}/omni-sense-iep4:${TAG}" 8004

# Get IEP URLs for EEP configuration
IEP2_URL=$(aws apprunner describe-service --service-arn "$(aws apprunner list-services --region ${REGION} --query 'ServiceSummaryList[?ServiceName==\`omni-sense-iep2\`].ServiceArn' --output text)" --region ${REGION} --query 'Service.ServiceUrl' --output text)
IEP3_URL=$(aws apprunner describe-service --service-arn "$(aws apprunner list-services --region ${REGION} --query 'ServiceSummaryList[?ServiceName==\`omni-sense-iep3\`].ServiceArn' --output text)" --region ${REGION} --query 'Service.ServiceUrl' --output text)
IEP4_URL=$(aws apprunner describe-service --service-arn "$(aws apprunner list-services --region ${REGION} --query 'ServiceSummaryList[?ServiceName==\`omni-sense-iep4\`].ServiceArn' --output text)" --region ${REGION} --query 'Service.ServiceUrl' --output text)

deploy_apprunner "omni-sense-eep" "${REGISTRY}/omni-sense-eep:${TAG}" 8000

# Update EEP with downstream URLs
EEP_ARN=$(aws apprunner list-services --region ${REGION} --query 'ServiceSummaryList[?ServiceName==\`omni-sense-eep\`].ServiceArn' --output text)
aws apprunner update-service \
  --service-arn "${EEP_ARN}" \
  --source-configuration "ImageRepository={ImageIdentifier=${REGISTRY}/omni-sense-eep:${TAG},ImageRepositoryType=ECR},AutoDeploymentsEnabled=false,ImageConfiguration={Port=8000,RuntimeEnvironmentVariables={OMNI_IEP2_URL=${IEP2_URL},OMNI_IEP3_URL=${IEP3_URL},OMNI_IEP4_URL=${IEP4_URL},OMNI_RATE_LIMIT=5/minute}}}" \
  --region "${REGION}" \
  --no-cli-pager

echo ""
echo "✅ Deployment initiated! Service URLs (may take 2-3 min to be ready):"
echo "  EEP:  https://$(aws apprunner describe-service --service-arn ${EEP_ARN} --region ${REGION} --query 'Service.ServiceUrl' --output text)"
echo "  IEP2: https://${IEP2_URL}"
echo "  IEP3: https://${IEP3_URL}"
echo "  IEP4: https://${IEP4_URL}"
