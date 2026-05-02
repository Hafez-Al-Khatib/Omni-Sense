#!/bin/bash
# ============================================================================
# Omni-Sense Hetzner CX23 Bootstrap
# ============================================================================
# Run this script ON the Hetzner server after SSH login.
#
# TWO PATHS:
#   1. GHCR  (FAST ~2 min) — pulls pre-built images from GitHub Container
#      Registry. Requires a GitHub PAT with `read:packages`.
#   2. BUILD (SLOW ~15 min) — builds all images on the server. Slower but
#      requires no external auth and guarantees the latest code.
#
# Quick start:
#   ssh root@<server-ip>
#   bash <(curl -fsSL https://raw.githubusercontent.com/YOUR_USER/omni-sense/main/scripts/bootstrap-hetzner.sh)
# ============================================================================

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Hafez-Al-Khatib/Omni-Sense.git}"
OMNI_DOMAIN="${OMNI_DOMAIN:-}"
GHCR_USER="${GHCR_USER:-}"
GITHUB_PAT="${GITHUB_PAT:-}"
DEPLOY_MODE="${DEPLOY_MODE:-}"

echo "========================================"
echo "  Omni-Sense Hetzner Bootstrap"
echo "========================================"
echo ""

# ─── 1. Validate ────────────────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Must run as root (or with sudo)."
    exit 1
fi

if [ -z "$OMNI_DOMAIN" ]; then
    echo -n "Enter your DuckDNS domain (without .duckdns.org): "
    read OMNI_DOMAIN
fi
FULL_DOMAIN="${OMNI_DOMAIN}.duckdns.org"

# Choose deploy mode
if [ -z "$DEPLOY_MODE" ]; then
    echo ""
    echo "Choose deployment mode:"
    echo "  [1] GHCR  — pull pre-built images (~2 min, needs GitHub PAT)"
    echo "  [2] BUILD — build images on server (~15 min, no auth needed)"
    echo -n "Select (1/2): "
    read choice
    case "$choice" in
        1) DEPLOY_MODE="ghcr" ;;
        *) DEPLOY_MODE="build" ;;
    esac
fi

if [ "$DEPLOY_MODE" = "ghcr" ]; then
    if [ -z "$GHCR_USER" ]; then
        echo -n "Enter your GitHub username: "
        read GHCR_USER
    fi
    if [ -z "$GITHUB_PAT" ]; then
        echo -n "Enter GitHub Personal Access Token (read:packages): "
        read -s GITHUB_PAT
        echo ""
    fi
fi

echo ""
echo "Domain:     $FULL_DOMAIN"
echo "Mode:       $DEPLOY_MODE"
[ "$DEPLOY_MODE" = "ghcr" ] && echo "GHCR User:  $GHCR_USER"
echo ""

# ─── 2. System update & Docker install ──────────────────────────────────────
echo "[1/7] Updating system and installing Docker..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    ca-certificates curl gnupg git jq ufw \
    apt-transport-https software-properties-common

# Docker official repo
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable docker
systemctl start docker

# ─── 3. Firewall ────────────────────────────────────────────────────────────
echo ""
echo "[2/7] Configuring UFW firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 1883/tcp
ufw allow 8883/tcp
ufw --force enable

# ─── 4. Clone repo ──────────────────────────────────────────────────────────
echo ""
echo "[3/7] Cloning repository..."
PROJECT_DIR="/opt/omni-sense"
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Directory exists. Pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# ─── 5. Generate secrets ────────────────────────────────────────────────────
echo ""
echo "[4/7] Generating secrets..."

GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 24)
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 24)
MQTT_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 24)
PROM_PASS=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 24)
MLFLOW_PASS=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 24)

echo "  Hashing passwords with Caddy..."
PROM_HASH=$(docker run --rm caddy:2 caddy hash-password --plaintext "$PROM_PASS")
MLFLOW_HASH=$(docker run --rm caddy:2 caddy hash-password --plaintext "$MLFLOW_PASS")

# ─── 6. Write .env.production ───────────────────────────────────────────────
echo ""
echo "[5/7] Writing .env.production..."
cat > .env.production <<EOF
OMNI_DOMAIN=${FULL_DOMAIN}
OMNI_PROM_BASIC_HASH=${PROM_HASH}
OMNI_MLFLOW_BASIC_HASH=${MLFLOW_HASH}
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
POSTGRES_USER=omni
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=omnisense
MQTT_PASSWORD=${MQTT_PASSWORD}
OMNI_RATE_LIMIT=100/minute
EOF
chmod 600 .env.production

# ─── 7. Deploy ──────────────────────────────────────────────────────────────
COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

if [ "$DEPLOY_MODE" = "ghcr" ]; then
    echo ""
    echo "[6/7] Logging in to GHCR and pulling images..."
    echo "$GITHUB_PAT" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
    export GHCR_USER
    COMPOSE_CMD="$COMPOSE_CMD -f docker-compose.ghcr.yml"
    echo ""
    echo "Pulling images..."
    $COMPOSE_CMD pull
    echo ""
    echo "Starting services (no-build)..."
    $COMPOSE_CMD up -d --no-build
else
    echo ""
    echo "[6/7] Building Docker images (~10-15 min on CX23)..."
    $COMPOSE_CMD build --parallel
    echo ""
    echo "Starting services..."
    $COMPOSE_CMD up -d
fi

# ─── 8. Wait for health ─────────────────────────────────────────────────────
echo ""
echo "[7/7] Waiting for services to become healthy (max 120s)..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    HEALTHY=$(docker compose ps --format json 2>/dev/null | jq -r '[.[] | select(.Health == "healthy")] | length' 2>/dev/null || echo "0")
    TOTAL=$(docker compose ps --format json 2>/dev/null | jq -r 'length' 2>/dev/null || echo "0")
    echo "  $HEALTHY / $TOTAL services healthy..."
    if [ "$HEALTHY" -eq "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

# ─── 9. Quick smoke test ────────────────────────────────────────────────────
echo ""
echo "Running quick smoke test..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://eep.${FULL_DOMAIN}/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  EEP health check: OK (200)"
else
    echo "  EEP health check: FAIL ($HTTP_CODE) — DNS/TLS may still be propagating"
fi

# ─── 10. Print credentials ──────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  DEPLOYMENT COMPLETE"
echo "========================================"
echo ""
echo "Public URLs:"
echo "  API Gateway:  https://eep.${FULL_DOMAIN}"
echo "  Grafana:      https://grafana.${FULL_DOMAIN}"
echo "  Prometheus:   https://prometheus.${FULL_DOMAIN}"
echo "  MLflow:       https://mlflow.${FULL_DOMAIN}"
echo ""
echo "Credentials (SAVE THESE):"
echo "  Grafana:      admin / ${GRAFANA_ADMIN_PASSWORD}"
echo "  Prometheus:   admin / ${PROM_PASS}"
echo "  MLflow:       admin / ${MLFLOW_PASS}"
echo "  Postgres:     omni / ${POSTGRES_PASSWORD}"
echo "  MQTT:         omni / ${MQTT_PASSWORD}"
echo ""
echo "Test from your laptop:"
echo "  OMNI_URL=https://eep.${FULL_DOMAIN} python test_public_url.py"
echo ""

# Write credentials to file
cat > /root/omni-credentials.txt <<EOF
Omni-Sense Deployment Credentials
Generated: $(date -Iseconds)
Domain: ${FULL_DOMAIN}

API Gateway:  https://eep.${FULL_DOMAIN}
Grafana:      https://grafana.${FULL_DOMAIN}
Prometheus:   https://prometheus.${FULL_DOMAIN}
MLflow:       https://mlflow.${FULL_DOMAIN}

Grafana:
  User:     admin
  Password: ${GRAFANA_ADMIN_PASSWORD}

Prometheus:
  User:     admin
  Password: ${PROM_PASS}

MLflow:
  User:     admin
  Password: ${MLFLOW_PASS}

Postgres:
  User:     omni
  Password: ${POSTGRES_PASSWORD}

MQTT:
  User:     omni
  Password: ${MQTT_PASSWORD}
EOF
chmod 600 /root/omni-credentials.txt
echo "Credentials saved to: /root/omni-credentials.txt"
echo ""
echo "========================================"
