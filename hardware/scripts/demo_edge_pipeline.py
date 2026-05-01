#!/usr/bin/env python3
"""Omni-Sense Edge Pipeline Demo & Integration Test

This script:
  1. Starts a local MQTT subscriber on omni/diagnosis/+/+
  2. Launches the RPi edge agent in SIMULATE_MODE for one inference cycle
  3. Waits for the diagnosis JSON message
  4. Validates that the result contains expected keys (label, confidence, is_ood)
  5. Prints a human-readable summary

Usage:
  cd omni-sense
  python hardware/scripts/demo_edge_pipeline.py

Requirements:
  - paho-mqtt, numpy, scipy, onnxruntime installed
  - IEP2 ONNX models present at iep2/models/
  - A running Mosquitto broker on localhost:1883 (or override via env)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

# ─── Configuration ────────────────────────────────────────────────────────────

MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
DIAGNOSIS_TOPIC = "omni/diagnosis/#"
TIMEOUT_S = 90.0

# ─── MQTT Subscriber ──────────────────────────────────────────────────────────

results: list[dict] = []
result_event = threading.Event()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[demo] MQTT subscriber connected to {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(DIAGNOSIS_TOPIC)
        print(f"[demo] Subscribed to {DIAGNOSIS_TOPIC}")
    else:
        print(f"[demo] MQTT connect failed rc={rc}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[demo] JSON decode error: {exc}")
        return
    results.append(payload)
    result_event.set()
    print(f"[demo] Received diagnosis on {msg.topic}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def start_subscriber() -> mqtt.Client:
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()
    return client


def run_agent_once() -> subprocess.Popen:
    """Launch the edge agent in simulation mode for a single window."""
    agent_path = Path(__file__).resolve().parent.parent / "rpi_edge_agent" / "agent.py"
    env = os.environ.copy()
    env["SIMULATE_MODE"] = "1"
    env["SENSOR_ID"] = "S-DEMO-001"
    env["SITE_ID"] = "demo/lab"
    env["MQTT_HOST"] = MQTT_HOST
    env["MQTT_PORT"] = str(MQTT_PORT)
    env["VAD_THRESHOLD"] = "0.0001"  # low threshold so synthetic data always passes VAD

    print(f"[demo] Starting edge agent: {agent_path}")
    proc = subprocess.Popen(
        [sys.executable, str(agent_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def validate_result(payload: dict) -> bool:
    """Assert expected keys and sensible values."""
    diag = payload.get("diagnosis", {})
    required = ["label", "confidence", "is_ood"]
    for key in required:
        if key not in diag:
            print(f"[demo] VALIDATION FAIL: missing diagnosis.{key}")
            return False

    print(f"[demo] VALIDATION PASS")
    print(f"       label      = {diag['label']}")
    print(f"       confidence = {diag['confidence']}")
    print(f"       is_ood     = {diag['is_ood']}")
    print(f"       rms        = {payload.get('rms')}")
    print(f"       snr_db     = {payload.get('snr_db')}")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Omni-Sense Edge Pipeline Demo")
    print("=" * 60)

    # Check models exist
    model_dir = Path(__file__).resolve().parent.parent.parent / "iep2" / "models"
    for fname in ("isolation_forest.onnx", "xgboost_classifier.onnx"):
        if not (model_dir / fname).exists():
            print(f"[demo] ERROR: Model not found: {model_dir / fname}")
            print("[demo] Train or download models before running this demo.")
            return 1

    # Start subscriber
    sub = start_subscriber()
    time.sleep(1.0)  # let subscriber connect

    # Start agent
    proc = run_agent_once()

    print(f"[demo] Waiting up to {TIMEOUT_S}s for diagnosis message...")
    found = result_event.wait(timeout=TIMEOUT_S)

    # Clean up
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
    sub.loop_stop()
    sub.disconnect()

    if not found:
        print("[demo] ERROR: Timed out waiting for diagnosis message.")
        print("[demo] Make sure the Mosquitto broker is running on {}:{}".format(MQTT_HOST, MQTT_PORT))
        return 1

    payload = results[-1]
    print("\n[demo] Full payload:")
    print(json.dumps(payload, indent=2))

    if validate_result(payload):
        print("\n[demo] Edge pipeline demo completed successfully.")
        return 0
    else:
        print("\n[demo] Edge pipeline demo completed with validation errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
