"""
Paho MQTT client for Omni-Sense live API.
Subscribes to sensors/+/result, persists to TimescaleDB and memory cache.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger("mqtt_client")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MQTT_HOST = os.getenv("MQTT_HOST", "mqtt-broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "60"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "sensors/+/result")

# ---------------------------------------------------------------------------
# Shared state (protected by _lock)
# ---------------------------------------------------------------------------
_latest_result: dict[str, Any] = {}
_lock = threading.Lock()

_db_callback: Optional[Callable[..., Any]] = None


def get_latest_result() -> dict[str, Any]:
    """Return the newest inference result from memory."""
    with _lock:
        return dict(_latest_result)


def _set_latest_result(payload: dict[str, Any]) -> None:
    with _lock:
        _latest_result.clear()
        _latest_result.update(payload)


def register_db_callback(fn: Callable[..., Any]) -> None:
    global _db_callback
    _db_callback = fn


# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------
def on_connect(
    client: mqtt.Client,
    userdata: Any,
    flags: dict,
    rc: int,
    properties: Any = None,
) -> None:
    if rc == 0:
        logger.info("MQTT connected to %s:%s", MQTT_HOST, MQTT_PORT)
        client.subscribe(MQTT_TOPIC)
        logger.info("Subscribed to %s", MQTT_TOPIC)
    else:
        logger.error("MQTT connection failed, code %s", rc)


def on_disconnect(
    client: mqtt.Client, userdata: Any, rc: int
) -> None:
    if rc != 0:
        logger.warning("MQTT unexpected disconnect (rc=%s), auto-reconnect enabled", rc)


def on_message(
    client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage
) -> None:
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Invalid JSON on %s: %s", msg.topic, exc)
        return

    # Normalise timestamp
    ts_raw = payload.get("timestamp")
    if ts_raw is None:
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Derive sensor_id from topic if not present in payload
    # topic pattern: sensors/<sensor_id>/result
    if "sensor_id" not in payload:
        parts = msg.topic.split("/")
        if len(parts) >= 2:
            payload["sensor_id"] = parts[1]

    # Enrich payload for memory cache
    payload["received_at"] = datetime.now(timezone.utc).isoformat()

    # Update in-memory latest result
    _set_latest_result(payload)

    # Persist to DB via async callback (fire-and-forget into event loop)
    if _db_callback is not None:
        try:
            _db_callback(payload)
        except Exception as exc:
            logger.exception("DB callback failed: %s", exc)

    logger.debug("Processed message from %s", msg.topic)


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------
_client: Optional[mqtt.Client] = None


def start_mqtt(loop: Any = None) -> mqtt.Client:
    """Initialise and connect the MQTT client in a background thread."""
    global _client
    _client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    _client.on_connect = on_connect
    _client.on_disconnect = on_disconnect
    _client.on_message = on_message
    _client.reconnect_delay_set(min_delay=1, max_delay=30)

    try:
        _client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
    except Exception as exc:
        logger.error("Initial MQTT connect failed: %s", exc)

    _client.loop_start()
    return _client


def stop_mqtt() -> None:
    global _client
    if _client:
        _client.loop_stop()
        _client.disconnect()
        _client = None
        logger.info("MQTT client stopped")


def is_mqtt_connected() -> bool:
    return _client.is_connected() if _client else False
