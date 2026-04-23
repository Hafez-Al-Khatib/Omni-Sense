#!/usr/bin/env bash
# gen_certs.sh — Generate a fleet CA and device certificates for Omni-Sense.
#
# Usage:
#   ./omni/scripts/gen_certs.sh                           # CA + gateway
#   ./omni/scripts/gen_certs.sh S-HAMRA-001 S-HAMRA-002  # + sensor certs
#
# Output (all in ./certs/):
#   ca.key / ca.crt              — Fleet CA (keep ca.key offline in prod)
#   server.key / server.crt      — Mosquitto broker cert
#   gateway.key / gateway.crt    — MQTT gateway client cert
#   {sensor_id}.key / .crt       — Per-sensor client certs
#
# Requirements: openssl ≥ 1.1.1

set -euo pipefail
CERTS_DIR="certs"
mkdir -p "$CERTS_DIR"
DAYS=825   # ~2.25 years — Apple/Google max for TLS certs

echo "=== Omni-Sense PKI bootstrap ==="

# ── Fleet CA ──────────────────────────────────────────────────────────────
if [[ ! -f "$CERTS_DIR/ca.key" ]]; then
    openssl ecparam -genkey -name prime256v1 -out "$CERTS_DIR/ca.key"
    openssl req -new -x509 -days $DAYS \
        -key "$CERTS_DIR/ca.key" \
        -out "$CERTS_DIR/ca.crt" \
        -subj "/C=LB/O=Omni-Sense/CN=Omni-Sense Fleet CA"
    echo "✓ Fleet CA generated"
else
    echo "  Fleet CA already exists — skipping"
fi

# ── Broker cert ───────────────────────────────────────────────────────────
if [[ ! -f "$CERTS_DIR/server.crt" ]]; then
    openssl ecparam -genkey -name prime256v1 -out "$CERTS_DIR/server.key"
    openssl req -new \
        -key "$CERTS_DIR/server.key" \
        -out "$CERTS_DIR/server.csr" \
        -subj "/C=LB/O=Omni-Sense/CN=mqtt.omni-sense.lb"
    # SAN required for TLS hostname verification
    cat > "$CERTS_DIR/server_ext.cnf" <<EOF
[SAN]
subjectAltName=DNS:mqtt.omni-sense.lb,DNS:localhost,IP:127.0.0.1
EOF
    openssl x509 -req -days $DAYS \
        -in "$CERTS_DIR/server.csr" \
        -CA "$CERTS_DIR/ca.crt" \
        -CAkey "$CERTS_DIR/ca.key" \
        -CAcreateserial \
        -extfile "$CERTS_DIR/server_ext.cnf" \
        -extensions SAN \
        -out "$CERTS_DIR/server.crt"
    echo "✓ Broker cert generated"
fi

# ── Helper: issue a client cert ───────────────────────────────────────────
issue_client_cert() {
    local NAME="$1"
    if [[ -f "$CERTS_DIR/${NAME}.crt" ]]; then
        echo "  $NAME cert already exists — skipping"
        return
    fi
    openssl ecparam -genkey -name prime256v1 -out "$CERTS_DIR/${NAME}.key"
    openssl req -new \
        -key "$CERTS_DIR/${NAME}.key" \
        -out "$CERTS_DIR/${NAME}.csr" \
        -subj "/C=LB/O=Omni-Sense/CN=${NAME}"
    openssl x509 -req -days $DAYS \
        -in "$CERTS_DIR/${NAME}.csr" \
        -CA "$CERTS_DIR/ca.crt" \
        -CAkey "$CERTS_DIR/ca.key" \
        -CAcreateserial \
        -out "$CERTS_DIR/${NAME}.crt"
    rm "$CERTS_DIR/${NAME}.csr"
    echo "✓ Client cert issued for $NAME"
}

# ── Gateway + any extra sensors passed as args ────────────────────────────
issue_client_cert "gateway"
for SENSOR_ID in "$@"; do
    issue_client_cert "$SENSOR_ID"
done

echo ""
echo "PKI ready.  Deploy to Mosquitto container:"
echo "  docker cp certs/ <mosquitto_container>:/mosquitto/certs/"
echo ""
echo "Provision each sensor with:"
echo "  ca.crt  — fleet CA (trust anchor)"
echo "  <sensor_id>.crt + <sensor_id>.key  — device identity"
