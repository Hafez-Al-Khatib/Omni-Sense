#!/bin/bash
# flash_fleet.sh — Batch-flash multiple Omni-Sense ESP32 sensors
# =============================================================================
# Usage:
#   1. Connect sensor #1 via USB → run: ./flash_fleet.sh 01
#   2. Disconnect #1, connect #2 → run: ./flash_fleet.sh 02
#   3. Repeat for each sensor
#
# The script overrides SENSOR_ID at compile time so each sensor gets a unique
# MQTT topic without editing config.h.
#
# If you set SENSOR_ID="AUTO" in config.h, you can simply run:
#   pio run -t upload
# for every sensor and the ID is derived from the MAC address automatically.
# This script is only needed if you want human-readable sequential IDs.

set -e

ID=${1:-01}
SENSOR_ID="esp32-s3-${ID}"

echo "=============================================="
echo "Flashing sensor: ${SENSOR_ID}"
echo "=============================================="

cd "$(dirname "$0")/.."

pio run -t upload --build-flag "-DSENSOR_ID=\"${SENSOR_ID}\""

echo ""
echo "✅ Sensor ${SENSOR_ID} flashed successfully."
echo "   Unplug this sensor and plug in the next one."
