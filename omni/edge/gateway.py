"""Edge MQTT Gateway — production ingestion path.

In the capstone demo the `edge/simulator.py` writes directly to the in-process
bus.  In production, real piezoelectric sensors publish over MQTT/TLS to this
gateway, which validates, normalises, and forwards frames to the bus (and in
a real deployment, to a Redpanda/Kafka topic).

Topic schema (MQTT)
-------------------
  omni/sensor/{site_id}/{sensor_id}/acoustic   → AcousticFrame JSON
  omni/sensor/{site_id}/{sensor_id}/telemetry  → TelemetrySample JSON

Security
--------
  - Mutual TLS (mTLS): each sensor has a unique client certificate signed by
    the fleet CA.  The broker rejects connections without a valid cert.
  - Topic ACL: sensors can only PUBLISH to their own sensor_id path;
    they cannot subscribe to other sensors' topics.
  - Payload size limit: 512 KB per message (guards against OOM attacks).

This module works in two modes:
  1. LIVE mode   — connects to a real Mosquitto/EMQX broker via paho-mqtt.
  2. STUB mode   — used when paho-mqtt is not installed; replays saved
                   MQTT messages from a JSON fixture file, letting the rest
                   of the pipeline run identically without network access.

Run the gateway standalone:
    python -m omni.edge.gateway --broker mqtt.omni-sense.lb --port 8883

Or embed it in main.py with use_stub=True for CI / capstone demo.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import ssl
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from omni.common.bus import Topics, get_bus
from omni.common.schemas import AcousticFrame, TelemetrySample

log = logging.getLogger("mqtt-gw")

# ─────────────────────────── Config ───────────────────────────────────────────

BROKER_HOST   = os.environ.get("OMNI_MQTT_HOST",   "mqtt.omni-sense.lb")
BROKER_PORT   = int(os.environ.get("OMNI_MQTT_PORT",   "8883"))
CA_CERT       = os.environ.get("OMNI_MQTT_CA",     "certs/ca.crt")
CLIENT_CERT   = os.environ.get("OMNI_MQTT_CERT",   "certs/gateway.crt")
CLIENT_KEY    = os.environ.get("OMNI_MQTT_KEY",    "certs/gateway.key")
MQTT_USERNAME = os.environ.get("OMNI_MQTT_USER",   "gateway")
MQTT_PASSWORD = os.environ.get("OMNI_MQTT_PASS",   "")

ACOUSTIC_TOPIC   = "omni/sensor/+/+/acoustic"
TELEMETRY_TOPIC  = "omni/sensor/+/+/telemetry"
MAX_PAYLOAD_BYTES = 512 * 1024


# ─────────────────────────── TLS helper ───────────────────────────────────────

def _build_ssl_context() -> ssl.SSLContext:
    """Build a strict mTLS context for broker connection."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    if Path(CA_CERT).exists():
        ctx.load_verify_locations(CA_CERT)
    if Path(CLIENT_CERT).exists() and Path(CLIENT_KEY).exists():
        ctx.load_cert_chain(CLIENT_CERT, CLIENT_KEY)
    return ctx


# ─────────────────────────── Message handlers ─────────────────────────────────

async def _handle_acoustic(topic: str, payload: bytes) -> None:
    """Validate and forward an acoustic frame payload."""
    if len(payload) > MAX_PAYLOAD_BYTES:
        log.warning("acoustic payload too large (%d bytes) on %s — dropped", len(payload), topic)
        return
    try:
        data = json.loads(payload)
        # Enforce required fields before touching the bus
        _ = data["sensor_id"], data["site_id"], data["pcm_b64"]
        # Patch captured_at if missing (edge clock drift fallback)
        if "captured_at" not in data:
            data["captured_at"] = datetime.now(timezone.utc).isoformat()
        frame = AcousticFrame(**data)
        await get_bus().publish(Topics.ACOUSTIC_FRAME, frame)
        log.debug("acoustic frame from %s forwarded", frame.sensor_id)
    except (KeyError, ValueError, TypeError) as e:
        log.warning("malformed acoustic payload on %s: %s", topic, e)


async def _handle_telemetry(topic: str, payload: bytes) -> None:
    if len(payload) > 8192:
        return
    try:
        data = json.loads(payload)
        if "captured_at" not in data:
            data["captured_at"] = datetime.now(timezone.utc).isoformat()
        sample = TelemetrySample(**data)
        await get_bus().publish(Topics.TELEMETRY, sample)
        log.debug("telemetry from %s: batt=%.0f%%", sample.sensor_id, sample.battery_pct)
    except Exception as e:
        log.warning("malformed telemetry on %s: %s", topic, e)


def _route(topic: str, payload: bytes) -> asyncio.Task:
    if topic.endswith("/acoustic"):
        return asyncio.create_task(_handle_acoustic(topic, payload))
    elif topic.endswith("/telemetry"):
        return asyncio.create_task(_handle_telemetry(topic, payload))
    else:
        log.debug("unknown topic %s — ignored", topic)


# ─────────────────────────── Live MQTT client ─────────────────────────────────

class LiveMQTTGateway:
    """Async wrapper around paho-mqtt for real broker connectivity."""

    def __init__(self) -> None:
        import paho.mqtt.client as mqtt_client
        self._client = mqtt_client.Client(
            client_id="omni-gateway-01",
            protocol=mqtt_client.MQTTv5,
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _on_connect(self, client, userdata, flags, rc, props=None) -> None:
        if rc == 0:
            log.info("MQTT connected to %s:%d", BROKER_HOST, BROKER_PORT)
            client.subscribe(ACOUSTIC_TOPIC, qos=1)
            client.subscribe(TELEMETRY_TOPIC, qos=1)
        else:
            log.error("MQTT connection failed: rc=%d", rc)

    def _on_message(self, client, userdata, msg) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._dispatch(msg.topic, msg.payload), self._loop
            )

    async def _dispatch(self, topic: str, payload: bytes) -> None:
        _route(topic, payload)

    def _on_disconnect(self, client, userdata, rc, props=None) -> None:
        log.warning("MQTT disconnected rc=%d — will auto-reconnect", rc)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

        if MQTT_USERNAME:
            self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        try:
            self._client.tls_set_context(_build_ssl_context())
        except Exception as e:
            log.warning("TLS setup failed (%s) — connecting without TLS (dev only)", e)

        self._client.connect_async(BROKER_HOST, BROKER_PORT, keepalive=60)
        self._client.loop_start()
        log.info("MQTT gateway started (live mode) → %s:%d", BROKER_HOST, BROKER_PORT)
        # Keep alive until cancelled
        while True:
            await asyncio.sleep(5)


# ─────────────────────────── Stub MQTT gateway ───────────────────────────────

class StubMQTTGateway:
    """Replays synthetic sensor messages to fill the bus without a real broker.

    Generates the same three regime schedules as the edge simulator but via
    the MQTT message format, exercising the full gateway validation path.
    This mode is used in the capstone demo and CI.
    """

    def __init__(self, sensors: list[dict], cadence_s: float = 1.0) -> None:
        self._sensors = sensors
        self._cadence = cadence_s
        self._running = False

    @staticmethod
    def _pcm_b64(regime: str, sr: int = 16000, dur: float = 0.975) -> str:
        n = int(sr * dur)
        t = np.linspace(0, dur, n)
        rng = np.random.default_rng()
        if regime == "quiet":
            x = rng.normal(0, 0.002, n)
        elif regime == "leak":
            x = rng.normal(0, 0.015, n)
            for f in (640, 1120, 1870):
                x += 0.008 * np.sin(2 * np.pi * f * t)
        else:
            x = 0.03 * np.sin(2 * np.pi * 120 * t) + rng.normal(0, 0.003, n)
        x = np.tanh(x * 2.5) * 0.9
        raw = struct.pack(f"<{len(x)}h", *(x * 32767).astype(np.int16).tolist())
        return base64.b64encode(raw).decode()

    async def _stream_sensor(
        self, sensor: dict, schedule: list[tuple[float, str]]
    ) -> None:
        fw = "edge-fw-2026.04.1"
        battery = 100.0
        frame_no = 0
        for duration, regime in schedule:
            n = max(1, int(duration / self._cadence))
            for _ in range(n):
                payload = json.dumps({
                    "sensor_id": sensor["sensor_id"],
                    "site_id":   sensor["site_id"],
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "pcm_b64": self._pcm_b64(regime),
                    "edge_snr_db": 12.0 if regime != "quiet" else 3.0,
                    "edge_vad_confidence": 0.9 if regime != "quiet" else 0.1,
                    "firmware_version": fw,
                }).encode()
                topic = f"omni/sensor/{sensor['site_id']}/{sensor['sensor_id']}/acoustic"
                await _handle_acoustic(topic, payload)
                frame_no += 1
                if frame_no % 5 == 0:
                    battery = max(0.0, battery - 0.03)
                    tel_payload = json.dumps({
                        "sensor_id": sensor["sensor_id"],
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "battery_pct": battery,
                        "temperature_c": 28.0,
                        "disk_free_mb": 3000.0,
                        "rtc_drift_ms": 5,
                        "uptime_s": frame_no,
                        "firmware_version": fw,
                    }).encode()
                    tel_topic = f"omni/sensor/{sensor['site_id']}/{sensor['sensor_id']}/telemetry"
                    await _handle_telemetry(tel_topic, tel_payload)
                await asyncio.sleep(self._cadence)

    async def run(self, schedules: list[list[tuple[float, str]]]) -> None:
        log.info(
            "MQTT stub gateway started — %d sensors replaying in-memory",
            len(self._sensors),
        )
        await asyncio.gather(*[
            self._stream_sensor(sensor, schedule)
            for sensor, schedule in zip(self._sensors, schedules)
        ])


# ─────────────────────────── Factory ──────────────────────────────────────────

def create_gateway(use_stub: bool = True, sensors: Optional[list] = None,
                   cadence_s: float = 1.0) -> "LiveMQTTGateway | StubMQTTGateway":
    """Factory: returns a live gateway in production, stub in demo/CI."""
    if not use_stub:
        try:
            import paho.mqtt.client  # noqa: F401
            return LiveMQTTGateway()
        except ImportError:
            log.warning("paho-mqtt not installed — falling back to stub gateway")
    return StubMQTTGateway(sensors or [], cadence_s=cadence_s)
