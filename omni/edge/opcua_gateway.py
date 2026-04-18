"""OPC-UA SCADA Integration Gateway for Omni-Sense.

Reads real pressure and flow data from an OPC-UA server and publishes
ScadaReading events to Topics.SCADA_READING ("scada.reading.v1").

IEP2's SCADA consistency check subscribes to this topic and can cross-
reference SCADA pressure readings against acoustic leak detections — a
sudden pressure drop co-occurring with a leak signature significantly
increases detection confidence.

Usage
-----
  # Production (real OPC-UA server):
  export OPCUA_ENDPOINT="opc.tcp://scada.omni-sense.lb:4840"
  export OPCUA_NODE_IDS='{"S-HAMRA-001": ["ns=2;i=1001","ns=2;i=1002","ns=2;i=1003"]}'

  # Development / CI (stub mode — auto-selected when asyncua is absent):
  python -m omni.edge.opcua_gateway

Architecture
------------
  asyncua available  → real OPC-UA subscription + polling fallback
  asyncua absent     → stub mode: synthetic Beirut diurnal pressure model

Stub pressure model
-------------------
Beirut municipal water supply operates at 2-8 bar with a characteristic
diurnal pattern: pressure peaks near 07:00 (morning demand surge) and
again near 19:00 (evening return), with the network minimum around 03:00.
The model adds:
  - A base pressure of 5 bar (typical mid-range)
  - Sinusoidal diurnal variation ±2 bar
  - White noise ±0.1 bar (sensor + pipeline turbulence)
  - Occasional transient pressure spikes (valve operations): ±1 bar,
    modelled as rare Bernoulli events with Gaussian shape
  - Flow is proportional to inverse-pressure with Gaussian noise
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from datetime import datetime, timezone
from typing import Optional

from omni.common.bus import Topics, get_bus
from omni.common.schemas import ScadaReading

log = logging.getLogger("opcua-gw")

# ─────────────────────── Environment config ───────────────────────────────────

OPCUA_ENDPOINT: str = os.environ.get("OPCUA_ENDPOINT", "opc.tcp://localhost:4840")

def _parse_node_ids() -> dict[str, list[str]]:
    """Parse OPCUA_NODE_IDS JSON env var.

    Expected format:
      { "sensor_id": ["pressure_node_id", "flow_node_id", "temperature_node_id"] }
    Nodes beyond index 0 are optional; missing nodes return None for that field.
    """
    import json
    raw = os.environ.get("OPCUA_NODE_IDS", "")
    if not raw:
        return {}
    try:
        mapping = json.loads(raw)
        if isinstance(mapping, dict):
            return {str(k): [str(n) for n in v] for k, v in mapping.items()}
    except Exception:  # noqa: BLE001
        log.warning("OPCUA_NODE_IDS is not valid JSON — stub mode will be used")
    return {}


# Backoff config for reconnect logic
_BACKOFF_BASE_S: float = 1.0
_BACKOFF_MAX_S: float = 60.0
_BACKOFF_FACTOR: float = 2.0

# ─────────────────────── Stub pressure model ──────────────────────────────────

def _diurnal_pressure(hour_utc: float, sensor_id: str = "") -> float:
    """Return a physically realistic Beirut water network pressure (bar).

    Model:
      base_pressure = 5.0 bar
      diurnal = 1.8 * sin(2π(t - 7h) / 24h)  ← peaks at 07:00
              + 0.8 * sin(2π(t - 19h) / 12h)  ← secondary peak at 19:00
      noise   = N(0, 0.08) bar

    The sensor_id seeds a deterministic per-sensor offset (±0.3 bar) to
    simulate spatial pressure variation across the network.
    """
    # Deterministic per-sensor spatial offset based on sensor_id hash
    seed_offset = (hash(sensor_id) % 1000) / 1000.0 * 0.6 - 0.3  # ±0.3 bar

    # Primary diurnal peak at 07:00 UTC (approximately EEST-3 → local 10:00)
    primary = 1.8 * math.sin(2 * math.pi * (hour_utc - 7.0) / 24.0)
    # Secondary evening peak at 19:00 UTC
    secondary = 0.8 * math.sin(2 * math.pi * (hour_utc - 19.0) / 12.0)

    base = 5.0 + seed_offset + primary + secondary
    noise = random.gauss(0.0, 0.08)

    # Clamp to realistic Beirut network range
    return max(2.0, min(8.0, base + noise))


def _transient_spike(rng: random.Random) -> float:
    """Return a pressure transient (bar) with low probability.

    Models valve closures / openings: ~2% probability per sample of a
    ±1 bar spike. Direction is random; magnitude is Gaussian-shaped around 0.7 bar.
    """
    if rng.random() < 0.02:
        direction = rng.choice([-1, 1])
        magnitude = abs(rng.gauss(0.7, 0.2))
        return direction * min(magnitude, 1.2)
    return 0.0


def _stub_reading(sensor_id: str, node_ids: list[str]) -> ScadaReading:
    """Generate one synthetic ScadaReading for the Beirut water network stub."""
    now = datetime.now(timezone.utc)
    hour_utc = now.hour + now.minute / 60.0

    rng = random.Random()
    pressure = _diurnal_pressure(hour_utc, sensor_id)
    pressure += _transient_spike(rng)
    pressure = max(1.5, min(8.5, pressure))  # absolute physical clamp

    # Flow is loosely anti-correlated with pressure (high pressure → lower flow
    # velocity through Bernoulli) with realistic Beirut distribution (5-40 L/s)
    base_flow = 20.0 - (pressure - 5.0) * 2.5
    flow = max(2.0, base_flow + random.gauss(0.0, 1.5))

    # Water temperature: stable 18-24 °C with slow seasonal drift, stub uses
    # time-of-day variation as a proxy (cold at night, warm midday)
    temp = 21.0 + 2.5 * math.sin(2 * math.pi * (hour_utc - 14.0) / 24.0)
    temp += random.gauss(0.0, 0.3)

    return ScadaReading(
        sensor_id=sensor_id,
        captured_at=now,
        pressure_bar=round(pressure, 3),
        flow_lps=round(flow, 2),
        temperature_c=round(temp, 2),
        node_ids=node_ids,
        source="stub",
    )


# ─────────────────────── OPC-UA real client ───────────────────────────────────

class OpcUaScadaClient:
    """Connects to an OPC-UA server and publishes ScadaReading events to the bus.

    Operates in two sub-modes:
      1. Subscription mode — server pushes data-change notifications (preferred)
      2. Poll fallback     — periodic read if subscriptions fail

    Reconnect logic uses exponential backoff on any connection error.
    """

    def __init__(
        self,
        endpoint: str = OPCUA_ENDPOINT,
        node_ids: Optional[dict[str, list[str]]] = None,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._node_ids: dict[str, list[str]] = node_ids or _parse_node_ids()
        self._poll_interval_s = poll_interval_s
        self._client: Optional[object] = None  # asyncua.Client when live
        self._subscription: Optional[object] = None
        self._running: bool = False
        self._use_stub: bool = False
        self._poll_task: Optional[asyncio.Task] = None

        # Detect asyncua availability at init time
        try:
            import asyncua  # noqa: F401
            self._use_stub = False
        except ImportError:
            log.warning(
                "asyncua not installed — OPC-UA gateway running in stub mode. "
                "Install with: pip install asyncua"
            )
            self._use_stub = True

    # ── Public API ────────────────────────────────────────────────────────────

    async def poll_forever(self, interval_s: Optional[float] = None) -> None:
        """Periodic polling fallback.  Publishes readings at each interval."""
        interval = interval_s or self._poll_interval_s
        log.info("OPC-UA poll loop started (interval=%.1fs)", interval)
        self._running = True
        while self._running:
            await self._poll_once()
            await asyncio.sleep(interval)

    async def wire(self) -> None:
        """Attach to the bus and start polling / subscription in background.

        Called from the platform runner; does not block.
        """
        self._running = True
        if self._use_stub:
            log.info("OPC-UA stub gateway wired — Beirut diurnal model active")
            self._poll_task = asyncio.create_task(
                self.poll_forever(), name="opcua-stub-poll"
            )
        else:
            # Launch real gateway with reconnect wrapper
            self._poll_task = asyncio.create_task(
                self._connect_with_retry(), name="opcua-live"
            )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        await self._disconnect()

    # ── Internal stub helpers ─────────────────────────────────────────────────

    async def _poll_once(self) -> None:
        """Emit one ScadaReading per configured sensor."""
        sensors = self._node_ids or {
            # Default sensors matching omni/main.py SENSORS list
            "S-HAMRA-001":  ["stub:pressure", "stub:flow", "stub:temp"],
            "S-HAMRA-002":  ["stub:pressure", "stub:flow", "stub:temp"],
            "S-VERDUN-001": ["stub:pressure", "stub:flow", "stub:temp"],
        }
        bus = get_bus()
        for sensor_id, nodes in sensors.items():
            if self._use_stub:
                reading = _stub_reading(sensor_id, nodes)
            else:
                reading = await self._read_live(sensor_id, nodes)
                if reading is None:
                    continue
            await bus.publish(Topics.SCADA_READING, reading)
            log.debug(
                "SCADA %s → %.2f bar  %.1f L/s  (source=%s)",
                sensor_id,
                reading.pressure_bar,
                reading.flow_lps or 0.0,
                reading.source,
            )

    # ── Real OPC-UA implementation ────────────────────────────────────────────

    async def _connect_with_retry(self) -> None:
        """Main loop: connect, subscribe, poll on failure, reconnect on loss."""
        backoff = _BACKOFF_BASE_S
        while self._running:
            try:
                await self._connect()
                backoff = _BACKOFF_BASE_S  # reset on successful connect
                await self._subscribe_or_poll()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "OPC-UA connection lost (%s) — reconnecting in %.0fs",
                    exc,
                    backoff,
                )
                await self._disconnect()
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_FACTOR, _BACKOFF_MAX_S)

    async def _connect(self) -> None:
        """Open the OPC-UA session."""
        from asyncua import Client  # type: ignore[import]
        log.info("OPC-UA connecting to %s", self._endpoint)
        self._client = Client(url=self._endpoint)
        await self._client.connect()  # type: ignore[union-attr]
        log.info("OPC-UA connected to %s", self._endpoint)

    async def _disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    async def _subscribe_or_poll(self) -> None:
        """Attempt subscription; fall back to polling if it raises."""
        try:
            await self._start_subscription()
            # Keep alive — the subscription handler does all the work
            while self._running:
                await asyncio.sleep(5)
        except Exception as sub_exc:  # noqa: BLE001
            log.warning(
                "OPC-UA subscription failed (%s) — falling back to polling",
                sub_exc,
            )
            await self.poll_forever()

    async def _start_subscription(self) -> None:
        """Create an OPC-UA subscription and register data-change handlers."""
        from asyncua import ua  # type: ignore[import]

        handler = _DataChangeHandler(self)
        sub = await self._client.create_subscription(  # type: ignore[union-attr]
            period=int(self._poll_interval_s * 1000),  # ms
            handler=handler,
        )
        self._subscription = sub

        # Build a flat list of (sensor_id, field_index, node) tuples
        for sensor_id, nodes in self._node_ids.items():
            for idx, node_id_str in enumerate(nodes):
                node = self._client.get_node(node_id_str)  # type: ignore[union-attr]
                await sub.subscribe_data_change(node)
                log.debug(
                    "Subscribed to OPC-UA node %s (sensor=%s, field=%d)",
                    node_id_str,
                    sensor_id,
                    idx,
                )

        # Store sensor→node mapping in handler for reverse-lookup
        handler.node_ids = self._node_ids

    async def _read_live(
        self, sensor_id: str, nodes: list[str]
    ) -> Optional[ScadaReading]:
        """Read current values from OPC-UA nodes for one sensor."""
        if self._client is None:
            return None
        try:
            pressure: Optional[float] = None
            flow: Optional[float] = None
            temp: Optional[float] = None

            for idx, node_id_str in enumerate(nodes[:3]):
                node = self._client.get_node(node_id_str)  # type: ignore[union-attr]
                dv = await node.read_data_value()
                val = float(dv.Value.Value) if dv.Value.Value is not None else None
                if idx == 0:
                    pressure = val
                elif idx == 1:
                    flow = val
                elif idx == 2:
                    temp = val

            if pressure is None:
                return None

            return ScadaReading(
                sensor_id=sensor_id,
                captured_at=datetime.now(timezone.utc),
                pressure_bar=round(pressure, 3),
                flow_lps=round(flow, 2) if flow is not None else None,
                temperature_c=round(temp, 2) if temp is not None else None,
                node_ids=nodes,
                source="opcua",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("OPC-UA read failed for %s: %s", sensor_id, exc)
            return None


class _DataChangeHandler:
    """asyncua subscription handler — receives data-change notifications."""

    def __init__(self, gateway: OpcUaScadaClient) -> None:
        self._gw = gateway
        # node_ids is set by _start_subscription after construction
        self.node_ids: dict[str, list[str]] = {}
        # Buffer partial readings until all fields arrive
        self._buffer: dict[str, dict[int, float]] = {}

    def datachange_notification(self, node, val, data) -> None:  # type: ignore[override]
        """Called by asyncua when a subscribed node value changes."""
        node_id_str = str(node.nodeid)
        sensor_id, field_idx = self._reverse_lookup(node_id_str)
        if sensor_id is None:
            return
        buf = self._buffer.setdefault(sensor_id, {})
        try:
            buf[field_idx] = float(val)
        except (TypeError, ValueError):
            return
        # Emit a reading whenever we have at least the pressure field (index 0)
        if 0 in buf:
            asyncio.get_event_loop().create_task(self._emit(sensor_id, buf.copy()))

    def _reverse_lookup(self, node_id_str: str) -> tuple[Optional[str], int]:
        for sensor_id, nodes in self.node_ids.items():
            for idx, nid in enumerate(nodes):
                if nid == node_id_str:
                    return sensor_id, idx
        return None, -1

    async def _emit(self, sensor_id: str, buf: dict[int, float]) -> None:
        reading = ScadaReading(
            sensor_id=sensor_id,
            captured_at=datetime.now(timezone.utc),
            pressure_bar=round(buf[0], 3),
            flow_lps=round(buf[1], 2) if 1 in buf else None,
            temperature_c=round(buf[2], 2) if 2 in buf else None,
            node_ids=self.node_ids.get(sensor_id, []),
            source="opcua",
        )
        await get_bus().publish(Topics.SCADA_READING, reading)
        log.debug(
            "SCADA[sub] %s → %.2f bar", sensor_id, reading.pressure_bar
        )


# ─────────────────────── Module-level singleton ───────────────────────────────

_gateway: Optional[OpcUaScadaClient] = None


def get_gateway() -> OpcUaScadaClient:
    global _gateway
    if _gateway is None:
        _gateway = OpcUaScadaClient()
    return _gateway


async def wire_async() -> None:
    """Async wire — call from an async context (e.g. platform startup)."""
    await get_gateway().wire()


def wire() -> None:
    """Schedule the OPC-UA gateway to start on the running event loop.

    Safe to call from synchronous code (e.g. main.py wire_everything()).
    If no event loop is running the gateway is marked pending and must be
    started explicitly via asyncio.run(wire_async()).
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(get_gateway().wire(), name="opcua-gw-wire")
        log.info("OPC-UA gateway scheduled on running event loop")
    except RuntimeError:
        # No running loop — caller must invoke wire_async() inside asyncio.run()
        log.info(
            "OPC-UA gateway registered (no running loop — call wire_async() "
            "inside your async startup coroutine)"
        )


# ─────────────────────── Standalone entry point ───────────────────────────────

async def _run_standalone() -> None:
    """Run the gateway standalone for testing / diagnostics."""
    import signal

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-5s %(name)-12s │ %(message)s",
    )

    gw = OpcUaScadaClient()

    def _on_sigint(_sig, _frame):
        log.info("shutdown requested")
        gw._running = False

    signal.signal(signal.SIGINT, _on_sigint)

    # Wire bus subscriber to print readings
    def _print_reading(payload: dict) -> None:  # type: ignore[override]
        print(
            f"  [{payload['sensor_id']}] "
            f"p={payload['pressure_bar']:.2f} bar  "
            f"q={payload.get('flow_lps', 'N/A')} L/s  "
            f"T={payload.get('temperature_c', 'N/A')}°C  "
            f"source={payload['source']}"
        )

    async def _async_print(payload: dict) -> None:
        _print_reading(payload)

    get_bus().subscribe(Topics.SCADA_READING, _async_print)
    bus_task = asyncio.create_task(get_bus().run(), name="bus")

    print(f"OPC-UA gateway standalone — endpoint: {gw._endpoint}")
    print("Press Ctrl-C to stop.\n")

    await gw.wire()
    await asyncio.sleep(30)  # run for 30 s then exit
    await gw.stop()
    get_bus().stop()
    bus_task.cancel()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
