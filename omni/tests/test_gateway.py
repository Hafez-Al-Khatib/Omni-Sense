"""Edge gateway: payload validation, topic routing, bus forwarding."""
from __future__ import annotations

import asyncio
import base64
import json
import struct
from datetime import datetime, timezone

import numpy as np
import pytest

from omni.edge.gateway import (
    _handle_acoustic,
    _handle_telemetry,
    _route,
    StubMQTTGateway,
    MAX_PAYLOAD_BYTES,
)
from omni.common.bus import InMemoryBus, Topics
import omni.common.bus as bus_mod


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_bus():
    bus_mod._bus = InMemoryBus()
    yield
    bus_mod._bus = None


@pytest.fixture()
def running_bus():
    """Async context: returns (bus, task) — caller must stop/cancel."""
    return bus_mod.get_bus()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pcm_b64(n_samples: int = 15600) -> str:
    x = np.zeros(n_samples, dtype=np.int16)
    return base64.b64encode(struct.pack(f"<{n_samples}h", *x.tolist())).decode()


def _acoustic_payload(
    sensor_id: str = "S-TEST-01",
    site_id: str   = "HAMRA",
    pcm_b64: str   = None,
) -> bytes:
    return json.dumps({
        "sensor_id":           sensor_id,
        "site_id":             site_id,
        "captured_at":         datetime.now(timezone.utc).isoformat(),
        "pcm_b64":             pcm_b64 or _pcm_b64(),
        "edge_snr_db":         14.0,
        "edge_vad_confidence": 0.85,
        "firmware_version":    "edge-fw-2026.04.1",
    }).encode()


def _telemetry_payload(
    sensor_id: str = "S-TEST-01",
    battery:   float = 85.0,
) -> bytes:
    return json.dumps({
        "sensor_id":       sensor_id,
        "captured_at":     datetime.now(timezone.utc).isoformat(),
        "battery_pct":     battery,
        "temperature_c":   27.5,
        "disk_free_mb":    3000.0,
        "rtc_drift_ms":    3,
        "uptime_s":        3600,
        "firmware_version": "edge-fw-2026.04.1",
    }).encode()


async def _drain(bus: InMemoryBus, n_messages: int = 1, timeout: float = 1.0) -> None:
    """Run the bus until n_messages have been dispatched or timeout expires."""
    task = asyncio.create_task(bus.run())
    try:
        await asyncio.wait_for(asyncio.sleep(timeout), timeout=timeout + 0.1)
    except asyncio.TimeoutError:
        pass
    bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── Acoustic handler ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_acoustic_frame_published_to_bus():
    received: list = []
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, lambda f: received.append(f) or asyncio.coroutine(lambda: None)())

    # Use a custom handler that's actually async
    async def capture(frame):
        received.append(frame)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, capture)

    topic = "omni/sensor/HAMRA/S-TEST-01/acoustic"
    await _handle_acoustic(topic, _acoustic_payload())
    await _drain(bus)

    assert len(received) == 1
    assert received[0]["sensor_id"] == "S-TEST-01"
    assert received[0]["site_id"]   == "HAMRA"


@pytest.mark.asyncio
async def test_oversized_acoustic_payload_is_dropped():
    received: list = []

    async def capture(frame):
        received.append(frame)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, capture)

    giant = b"x" * (MAX_PAYLOAD_BYTES + 1)
    await _handle_acoustic("omni/sensor/HAMRA/S-X/acoustic", giant)
    # No publish call was made, queue is empty — drain returns immediately
    await _drain(bus, timeout=0.05)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_malformed_acoustic_json_is_dropped():
    received: list = []

    async def capture(frame):
        received.append(frame)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, capture)

    await _handle_acoustic("omni/sensor/HAMRA/S-X/acoustic", b"not json {{}")
    await _drain(bus, timeout=0.05)
    assert len(received) == 0


@pytest.mark.asyncio
async def test_acoustic_missing_required_field_is_dropped():
    """Payload missing pcm_b64 must be silently dropped."""
    received: list = []

    async def capture(frame):
        received.append(frame)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, capture)

    bad = json.dumps({
        "sensor_id": "S-X",
        "site_id":   "HAMRA",
        # pcm_b64 deliberately omitted
    }).encode()
    await _handle_acoustic("omni/sensor/HAMRA/S-X/acoustic", bad)
    await _drain(bus, timeout=0.05)
    assert len(received) == 0


@pytest.mark.asyncio
async def test_acoustic_captured_at_patched_when_missing():
    """Gateway must inject captured_at if the edge clock hasn't synced."""
    received: list = []

    async def capture(frame):
        received.append(frame)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, capture)

    payload = json.dumps({
        "sensor_id": "S-TEST-02",
        "site_id":   "VERDUN",
        "pcm_b64":   _pcm_b64(),
        "edge_snr_db": 10.0,
        "edge_vad_confidence": 0.7,
        "firmware_version": "edge-fw-2026.04.1",
        # captured_at intentionally absent
    }).encode()
    await _handle_acoustic("omni/sensor/VERDUN/S-TEST-02/acoustic", payload)
    await _drain(bus)

    assert len(received) == 1
    assert received[0]["captured_at"] is not None


# ── Telemetry handler ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_telemetry_published_to_bus():
    received: list = []

    async def capture(sample):
        received.append(sample)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.TELEMETRY, capture)

    await _handle_telemetry("omni/sensor/HAMRA/S-TEST-01/telemetry", _telemetry_payload())
    await _drain(bus)

    assert len(received) == 1
    assert received[0]["sensor_id"]   == "S-TEST-01"
    assert received[0]["battery_pct"] == pytest.approx(85.0)


@pytest.mark.asyncio
async def test_malformed_telemetry_is_dropped():
    received: list = []

    async def capture(sample):
        received.append(sample)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.TELEMETRY, capture)

    await _handle_telemetry("omni/sensor/HAMRA/S-X/telemetry", b"garbage")
    await _drain(bus, timeout=0.05)
    assert len(received) == 0


# ── Router ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_dispatches_acoustic():
    received: list = []

    async def capture(frame):
        received.append(frame)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, capture)

    task = _route("omni/sensor/HAMRA/S-TEST-01/acoustic", _acoustic_payload())
    assert task is not None
    await task
    await _drain(bus)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_route_dispatches_telemetry():
    received: list = []

    async def capture(sample):
        received.append(sample)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.TELEMETRY, capture)

    task = _route("omni/sensor/HAMRA/S-TEST-01/telemetry", _telemetry_payload())
    await task
    await _drain(bus)
    assert len(received) == 1


def test_route_returns_none_for_unknown_topic():
    result = _route("omni/sensor/HAMRA/S-X/unknown", b"{}")
    assert result is None


# ── StubMQTTGateway ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stub_gateway_publishes_acoustic_frames():
    acoustic_frames: list = []

    async def cap_a(frame):
        acoustic_frames.append(frame)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, cap_a)
    bus_task = asyncio.create_task(bus.run())

    sensors = [{"sensor_id": "S-STUB-01", "site_id": "HAMRA"}]
    # cadence_s=0.001; duration = n * cadence_s → exactly n frames
    gw = StubMQTTGateway(sensors, cadence_s=0.001)
    schedule = [[(0.003, "quiet")]]   # n = int(0.003 / 0.001) = 3 frames
    await gw.run(schedule)
    await asyncio.sleep(0.1)   # drain bus

    bus.stop()
    bus_task.cancel()
    try:
        await bus_task
    except asyncio.CancelledError:
        pass

    assert len(acoustic_frames) == 3
    assert all(f["sensor_id"] == "S-STUB-01" for f in acoustic_frames)


@pytest.mark.asyncio
async def test_stub_gateway_emits_telemetry_every_5_frames():
    acoustic_frames: list = []
    telemetry_samples: list = []

    async def cap_a(f):
        acoustic_frames.append(f)

    async def cap_t(s):
        telemetry_samples.append(s)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, cap_a)
    bus.subscribe(Topics.TELEMETRY, cap_t)
    bus_task = asyncio.create_task(bus.run())

    sensors = [{"sensor_id": "S-STUB-02", "site_id": "ACHRAFIEH"}]
    gw = StubMQTTGateway(sensors, cadence_s=0.001)
    schedule = [[(0.010, "leak")]]   # n = int(0.010 / 0.001) = 10 frames
    await gw.run(schedule)
    await asyncio.sleep(0.1)

    bus.stop()
    bus_task.cancel()
    try:
        await bus_task
    except asyncio.CancelledError:
        pass

    assert len(acoustic_frames) == 10
    assert len(telemetry_samples) == 2


@pytest.mark.asyncio
async def test_stub_gateway_pcm_b64_is_valid_base64():
    for regime in ("quiet", "leak", "pump"):
        b64 = StubMQTTGateway._pcm_b64(regime)
        raw = base64.b64decode(b64)
        assert len(raw) % 2 == 0   # int16 pairs
        assert len(raw) // 2 > 0


@pytest.mark.asyncio
async def test_stub_gateway_multi_sensor():
    """Two sensors run concurrently — frames from both reach the bus."""
    frames: list = []

    async def cap(f):
        frames.append(f)

    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus.subscribe(Topics.ACOUSTIC_FRAME, cap)
    bus_task = asyncio.create_task(bus.run())

    sensors = [
        {"sensor_id": "S-A", "site_id": "HAMRA"},
        {"sensor_id": "S-B", "site_id": "HAMRA"},
    ]
    gw = StubMQTTGateway(sensors, cadence_s=0.001)
    schedules = [[(0.002, "quiet")], [(0.002, "leak")]]  # 2 frames each
    await gw.run(schedules)
    await asyncio.sleep(0.1)

    bus.stop()
    bus_task.cancel()
    try:
        await bus_task
    except asyncio.CancelledError:
        pass

    sensor_ids = {f["sensor_id"] for f in frames}
    assert "S-A" in sensor_ids
    assert "S-B" in sensor_ids
    assert len(frames) == 4
