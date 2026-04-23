#!/usr/bin/env bash
# Omni-Sense Raspberry Pi Edge Agent — systemd service installer
# Run as root:  sudo bash install.sh
set -euo pipefail

AGENT_USER="omni"
AGENT_DIR="/opt/omni-edge"
CERT_DIR="/etc/omni/certs"
SERVICE_NAME="omni-edge"
PYTHON_BIN="/usr/bin/python3"

echo "=== Omni-Sense Edge Agent Installer ==="

# ── 1. Check we are root ──────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Run this script as root: sudo bash install.sh"
  exit 1
fi

# ── 2. System dependencies ────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    i2c-tools \
    chrony \
    curl

# Enable I2C on Raspberry Pi (idempotent)
if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null && \
   ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
  # Try Raspberry Pi OS Bookworm path first
  BOOT_CFG="/boot/firmware/config.txt"
  [[ -f "$BOOT_CFG" ]] || BOOT_CFG="/boot/config.txt"
  echo "dtparam=i2c_arm=on" >> "$BOOT_CFG"
  echo "  I2C enabled in $BOOT_CFG — a reboot will be required."
fi

# ── 3. Create dedicated service user ─────────────────────────────────────────
echo "[2/7] Creating system user '$AGENT_USER'..."
id -u "$AGENT_USER" &>/dev/null || useradd -r -s /usr/sbin/nologin -d "$AGENT_DIR" "$AGENT_USER"
# Add to i2c group for hardware access
usermod -aG i2c "$AGENT_USER" 2>/dev/null || true

# ── 4. Install agent code ─────────────────────────────────────────────────────
echo "[3/7] Installing agent to $AGENT_DIR..."
install -d -o "$AGENT_USER" -g "$AGENT_USER" "$AGENT_DIR"
install -m 0644 -o "$AGENT_USER" -g "$AGENT_USER" \
    "$(dirname "$0")/agent.py" \
    "$(dirname "$0")/requirements.txt" \
    "$AGENT_DIR/"

# ── 5. Python virtual environment + pip deps ─────────────────────────────────
echo "[4/7] Setting up Python virtual environment..."
if [[ ! -d "$AGENT_DIR/venv" ]]; then
  sudo -u "$AGENT_USER" "$PYTHON_BIN" -m venv "$AGENT_DIR/venv"
fi
sudo -u "$AGENT_USER" "$AGENT_DIR/venv/bin/pip" install --quiet \
    --upgrade pip wheel
sudo -u "$AGENT_USER" "$AGENT_DIR/venv/bin/pip" install --quiet \
    -r "$AGENT_DIR/requirements.txt"

echo "  Python packages installed."

# ── 6. Certificate directory ──────────────────────────────────────────────────
echo "[5/7] Preparing certificate directory $CERT_DIR..."
install -d -o "$AGENT_USER" -g "$AGENT_USER" -m 0750 "$CERT_DIR"

if [[ ! -f "$CERT_DIR/ca.crt" ]]; then
  echo ""
  echo "  ACTION REQUIRED: Copy your mTLS certificates to $CERT_DIR/"
  echo "    ca.crt              — Certificate Authority cert"
  echo "    \${SENSOR_ID}.crt   — Client certificate"
  echo "    \${SENSOR_ID}.key   — Client private key (chmod 0600)"
  echo ""
fi

# ── 7. Systemd service ────────────────────────────────────────────────────────
echo "[6/7] Installing systemd service..."
install -m 0644 "$(dirname "$0")/omni-edge.service" \
    "/etc/systemd/system/${SERVICE_NAME}.service"

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"

echo "[7/7] Starting service..."
systemctl restart "${SERVICE_NAME}.service"
sleep 2
systemctl status "${SERVICE_NAME}.service" --no-pager || true

echo ""
echo "=== Installation complete ==="
echo ""
echo "  Useful commands:"
echo "    journalctl -u ${SERVICE_NAME} -f        # follow logs"
echo "    systemctl status ${SERVICE_NAME}         # check status"
echo "    systemctl restart ${SERVICE_NAME}        # restart agent"
echo "    systemctl stop ${SERVICE_NAME}           # stop agent"
echo ""
echo "  Environment overrides (edit /etc/systemd/system/${SERVICE_NAME}.service):"
echo "    SENSOR_ID, SITE_ID, MQTT_HOST, MQTT_PORT, VAD_THRESHOLD"
echo ""
