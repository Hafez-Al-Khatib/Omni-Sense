"""Omni-Sense Ops Console — FastAPI + WebSocket server.

Serves dashboard.html at GET / and bridges the in-memory bus to all connected
WebSocket clients.  Drop-in replacement for the Streamlit ops console with
much lower overhead (no polling, no Tornado re-renders).

Run standalone:
  python -m omni.ops_console.ws_server           # port 8765
  PORT=9000 python -m omni.ops_console.ws_server

Docker (replace streamlit in compose):
  command: python -m omni.ops_console.ws_server

The server also wires the full platform (orchestrator, spatial fusion, alerts,
dispatch, CMMS, notify, audit) so it is a self-contained demo runner.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from omni.alerts import engine
from omni.audit import log as audit_log
from omni.cmms import service as cmms_service
from omni.common import store
from omni.common.bus import Topics, get_bus
from omni.common.schemas import (
    Alert,
    DetectionResult,
    LeakHypothesis,
    TelemetrySample,
    WorkOrder,
)
from omni.common.store import SensorTwin
from omni.dispatch import router
from omni.edge.simulator import run_sensor
from omni.eep import orchestrator
from omni.notify import service as notify_service
from omni.spatial import fusion

log = logging.getLogger("ws_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)-12s │ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# ─────────────────────────── Configuration ────────────────────────────────────

PORT: int = int(os.environ.get("PORT", "8765"))
HOST: str = os.environ.get("HOST", "0.0.0.0")
HEARTBEAT_S: float = 10.0
DASHBOARD_HTML: Path = Path(__file__).parent / "dashboard.html"

SENSORS = [
    dict(sensor_id="S-HAMRA-001",     site_id="beirut/hamra",     lat=33.8978, lon=35.4828),
    dict(sensor_id="S-HAMRA-002",     site_id="beirut/hamra",     lat=33.8985, lon=35.4845),
    dict(sensor_id="S-VERDUN-001",    site_id="beirut/verdun",    lat=33.8840, lon=35.4910),
    dict(sensor_id="S-ACHRAFIEH-001", site_id="beirut/achrafieh", lat=33.8890, lon=35.5230),
]

LEAK_SCHEDULE  = [(5.0, "quiet"), (15.0, "leak"), (5.0, "quiet")]
QUIET_SCHEDULE = [(25.0, "quiet")]

# ─────────────────────────── WebSocket hub ────────────────────────────────────

class ConnectionHub:
    """Thread-safe set of connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        log.info("ws_client_connected total=%d", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("ws_client_disconnected total=%d", len(self._clients))

    async def broadcast(self, payload: dict) -> None:
        """Fan-out a message to all connected clients; silently drop stale ones."""
        data = json.dumps(payload, default=_json_serial)
        dead: list[WebSocket] = []
        async with self._lock:
            clients = list(self._clients)

        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    @property
    def count(self) -> int:
        return len(self._clients)


hub = ConnectionHub()


def _json_serial(obj: Any) -> Any:
    """JSON serialiser for types pydantic doesn't handle inline."""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


# ─────────────────────────── Bus → WebSocket bridges ─────────────────────────

def _strip_pcm(d: dict) -> dict:
    """Remove pcm_b64 so we don't flood the dashboard with 40 kB of base64."""
    d.pop("pcm_b64", None)
    return d


async def _on_detection(payload: dict) -> None:
    payload = _strip_pcm(dict(payload))
    det = DetectionResult(**payload)
    msg = {
        "type": "detection",
        "sensor_id":     det.sensor_id,
        "site_id":       det.site_id,
        "captured_at":   det.captured_at.isoformat(),
        "fused_p_leak":  round(det.fused_p_leak, 4),
        "xgb_p_leak":    round(det.xgb_p_leak, 4),
        "rf_p_leak":     round(det.rf_p_leak, 4),
        "cnn_p_leak":    round(det.cnn_p_leak, 4) if det.cnn_p_leak is not None else None,
        "ood_score":     round(det.ood_score, 4),
        "is_leak":       det.is_leak,
        "is_ood":        det.is_ood,
        "top_shap":      det.top_shap_features[:3],
        "yamnet_class":  det.yamnet_top_class,
    }
    await hub.broadcast(msg)


async def _on_telemetry(payload: dict) -> None:
    tel = TelemetrySample(**payload)
    msg = {
        "type":           "telemetry",
        "sensor_id":      tel.sensor_id,
        "captured_at":    tel.captured_at.isoformat(),
        "battery_pct":    tel.battery_pct,
        "temperature_c":  tel.temperature_c,
        "disk_free_mb":   tel.disk_free_mb,
        "rtc_drift_ms":   tel.rtc_drift_ms,
        "uptime_s":       tel.uptime_s,
        "firmware_version": tel.firmware_version,
    }
    await hub.broadcast(msg)


async def _on_alert_new(payload: dict) -> None:
    alert = Alert(**payload)
    await hub.broadcast(_alert_to_msg(alert))


async def _on_alert_state(payload: dict) -> None:
    alert = Alert(**payload)
    await hub.broadcast(_alert_to_msg(alert))


def _alert_to_msg(alert: Alert) -> dict:
    return {
        "type":                "alert",
        "alert_id":            str(alert.alert_id),
        "hypothesis_id":       str(alert.hypothesis_id),
        "state":               alert.state.value,
        "severity":            alert.severity.value,
        "severity_score":      round(alert.severity_score, 1),
        "title":               alert.title,
        "summary":             alert.summary,
        "lat":                 alert.lat,
        "lon":                 alert.lon,
        "pipe_segment_id":     alert.pipe_segment_id,
        "estimated_loss_lph":  alert.estimated_loss_lph,
        "assigned_crew_id":    alert.assigned_crew_id,
        "sla_due_at":          alert.sla_due_at.isoformat() if alert.sla_due_at else None,
        "created_at":          alert.created_at.isoformat(),
        "updated_at":          alert.updated_at.isoformat(),
    }


async def _on_hypothesis(payload: dict) -> None:
    h = LeakHypothesis(**payload)
    msg = {
        "type":            "hypothesis",
        "hypothesis_id":   str(h.hypothesis_id),
        "lat":             h.lat,
        "lon":             h.lon,
        "uncertainty_m":   round(h.uncertainty_m, 1),
        "pipe_segment_id": h.pipe_segment_id,
        "confidence":      round(h.confidence, 4),
        "estimated_flow_lps": h.estimated_flow_lps,
        "created_at":      h.created_at.isoformat(),
        "n_sensors":       len(h.contributing_detection_ids),
    }
    await hub.broadcast(msg)


async def _on_work_order(payload: dict) -> None:
    wo = WorkOrder(**payload)
    msg = {
        "type":            "work_order",
        "work_order_id":   str(wo.work_order_id),
        "alert_id":        str(wo.alert_id),
        "status":          wo.status.value,
        "crew_id":         wo.crew_id,
        "eta_minutes":     wo.eta_minutes,
        "parts_required":  wo.parts_required,
        "notes":           wo.notes,
        "created_at":      wo.created_at.isoformat(),
        "completed_at":    wo.completed_at.isoformat() if wo.completed_at else None,
        "repair_cost_usd": wo.repair_cost_usd,
    }
    await hub.broadcast(msg)


def _wire_bus_handlers() -> None:
    """Subscribe our bridge handlers to the bus (idempotent if called once)."""
    bus = get_bus()
    bus.subscribe(Topics.DETECTION,  _on_detection)
    bus.subscribe(Topics.TELEMETRY,  _on_telemetry)
    bus.subscribe(Topics.ALERT_NEW,  _on_alert_new)
    bus.subscribe(Topics.ALERT_STATE, _on_alert_state)
    bus.subscribe(Topics.HYPOTHESIS, _on_hypothesis)
    bus.subscribe(Topics.WORK_ORDER, _on_work_order)
    log.info("bus_handlers_wired")


# ─────────────────────────── Heartbeat ───────────────────────────────────────

async def _heartbeat_loop() -> None:
    """Send a heartbeat to all clients every HEARTBEAT_S seconds."""
    while True:
        await asyncio.sleep(HEARTBEAT_S)
        ok, _ = audit_log.verify_chain()
        twins  = list(store.twins().twins.values())
        alerts = await store.alerts().list_all()
        active = sum(1 for a in alerts
                     if a.state.value in ("NEW","ACKNOWLEDGED","DISPATCHED","ON_SITE"))
        msg = {
            "type":          "heartbeat",
            "ts":            datetime.now(UTC).isoformat(),
            "ws_clients":    hub.count,
            "audit_ok":      ok,
            "audit_events":  len(audit_log.CHAIN),
            "sensors_total": len(twins),
            "sensors_healthy": sum(1 for t in twins if t.is_healthy),
            "active_alerts": active,
        }
        await hub.broadcast(msg)

        # Also broadcast current sensor twin states
        for twin in twins:
            await hub.broadcast({
                "type":          "sensor_twin",
                "sensor_id":     twin.sensor_id,
                "site_id":       twin.site_id,
                "lat":           twin.lat,
                "lon":           twin.lon,
                "battery_pct":   twin.battery_pct,
                "temperature_c": twin.temperature_c,
                "last_p_leak":   round(twin.last_p_leak, 4),
                "is_healthy":    twin.is_healthy,
                "firmware_version": twin.firmware_version,
                "last_seen":     twin.last_seen.isoformat() if twin.last_seen else None,
            })


# ─────────────────────────── Platform boot ───────────────────────────────────

_platform_started = False
_platform_lock = threading.Lock()


def _wire_platform() -> None:
    """Wire all platform services (idempotent)."""
    orchestrator.wire()
    fusion.wire()
    engine.wire()
    router.wire()
    cmms_service.wire()
    notify_service.wire()
    audit_log.wire()
    _wire_bus_handlers()


async def _seed_twins() -> None:
    for s in SENSORS:
        await store.twins().upsert_twin(
            SensorTwin(
                sensor_id=s["sensor_id"],
                site_id=s["site_id"],
                lat=s["lat"],
                lon=s["lon"],
            )
        )


async def _run_scenario() -> None:
    """Spin sensor simulators in an infinite loop."""
    async def sensor_loop(sid, site, lat, lon, schedule):
        while True:
            await run_sensor(sid, site, lat, lon, schedule, cadence_s=1.0)

    tasks = [
        asyncio.create_task(sensor_loop("S-HAMRA-001",     "beirut/hamra",     33.8978, 35.4828, LEAK_SCHEDULE)),
        asyncio.create_task(sensor_loop("S-HAMRA-002",     "beirut/hamra",     33.8985, 35.4845, LEAK_SCHEDULE)),
        asyncio.create_task(sensor_loop("S-VERDUN-001",    "beirut/verdun",    33.8840, 35.4910, QUIET_SCHEDULE)),
        asyncio.create_task(sensor_loop("S-ACHRAFIEH-001", "beirut/achrafieh", 33.8890, 35.5230, QUIET_SCHEDULE)),
    ]
    await asyncio.gather(*tasks)


# ─────────────────────────── FastAPI app ─────────────────────────────────────

app = FastAPI(
    title="Omni-Sense Ops Console",
    description="Real-time WebSocket bridge for the Omni-Sense platform",
    version="2.0.0",
)


@app.on_event("startup")
async def _on_startup() -> None:
    global _platform_started
    with _platform_lock:
        if _platform_started:
            return
        _platform_started = True

    log.info("platform_startup wiring services...")
    _wire_platform()
    await _seed_twins()
    bus = get_bus()

    # Run bus + sensors + heartbeat as background tasks
    asyncio.create_task(bus.run(), name="bus")
    asyncio.create_task(_run_scenario(), name="scenario")
    asyncio.create_task(_heartbeat_loop(), name="heartbeat")
    log.info("platform_ready ws_server port=%d", PORT)


# ── Dashboard HTML ────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    """Serve the single-file Mapbox dashboard."""
    if not DASHBOARD_HTML.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "dashboard.html not found — ensure it is in omni/ops_console/"},
        )
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/sensors")
async def api_sensors():
    """Current sensor twin states."""
    twins = list(store.twins().twins.values())
    return JSONResponse([
        {
            "sensor_id":    t.sensor_id,
            "site_id":      t.site_id,
            "lat":          t.lat,
            "lon":          t.lon,
            "battery_pct":  t.battery_pct,
            "temperature_c": t.temperature_c,
            "last_p_leak":  round(t.last_p_leak, 4),
            "is_healthy":   t.is_healthy,
            "firmware_version": t.firmware_version,
            "last_seen":    t.last_seen.isoformat() if t.last_seen else None,
        }
        for t in twins
    ])


@app.get("/api/alerts")
async def api_alerts(limit: int = 50, active_only: bool = False):
    """Recent alerts, optionally filtered to active states only."""
    alerts = await store.alerts().list_all()
    if active_only:
        alerts = [a for a in alerts
                  if a.state.value in ("NEW","ACKNOWLEDGED","DISPATCHED","ON_SITE")]
    return JSONResponse([_alert_to_msg(a) for a in alerts[:limit]])


@app.get("/api/metrics")
async def api_metrics():
    """Frame count, alert count, audit chain status."""
    twins   = list(store.twins().twins.values())
    alerts  = await store.alerts().list_all()
    wos     = await store.work_orders().list_all()
    ok, bad = audit_log.verify_chain()
    active  = sum(1 for a in alerts
                  if a.state.value in ("NEW","ACKNOWLEDGED","DISPATCHED","ON_SITE"))
    loss    = sum(a.estimated_loss_lph or 0 for a in alerts
                  if a.state.value in ("NEW","ACKNOWLEDGED","DISPATCHED","ON_SITE"))
    frames  = sum(len(dq) for dq in store.twins().detection_window.values())
    return JSONResponse({
        "sensors_total":    len(twins),
        "sensors_healthy":  sum(1 for t in twins if t.is_healthy),
        "frames_processed": frames,
        "hypotheses_total": len(store.hypotheses()),
        "alerts_total":     len(alerts),
        "alerts_active":    active,
        "work_orders_total": len(wos),
        "work_orders_closed": sum(1 for w in wos if w.completed_at is not None),
        "estimated_loss_lph": round(loss, 1),
        "audit_chain_ok":   ok,
        "audit_events":     len(audit_log.CHAIN),
        "audit_broken_at":  str(bad) if bad else None,
        "notifications_sent": len(notify_service.INBOX),
        "ws_clients":       hub.count,
        "ts":               datetime.now(UTC).isoformat(),
    })


@app.get("/api/hypotheses")
async def api_hypotheses(limit: int = 20):
    """Recent spatial hypotheses."""
    hyps = store.hypotheses()
    return JSONResponse([
        {
            "hypothesis_id":   str(h.hypothesis_id),
            "lat":             h.lat,
            "lon":             h.lon,
            "uncertainty_m":   h.uncertainty_m,
            "pipe_segment_id": h.pipe_segment_id,
            "confidence":      h.confidence,
            "estimated_flow_lps": h.estimated_flow_lps,
            "n_sensors":       len(h.contributing_detection_ids),
            "created_at":      h.created_at.isoformat(),
        }
        for h in hyps[-limit:]
    ])


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await hub.connect(websocket)

    # Send initial state snapshot so the client populates immediately
    await _send_initial_snapshot(websocket)

    try:
        while True:
            # Keep connection alive; client may send pings
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle client-sent ping
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except TimeoutError:
                # Send server-side ping
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(websocket)


async def _send_initial_snapshot(ws: WebSocket) -> None:
    """Push current state to a freshly connected client."""
    # Sensor twins
    for twin in store.twins().twins.values():
        await ws.send_text(json.dumps({
            "type":          "sensor_twin",
            "sensor_id":     twin.sensor_id,
            "site_id":       twin.site_id,
            "lat":           twin.lat,
            "lon":           twin.lon,
            "battery_pct":   twin.battery_pct,
            "temperature_c": twin.temperature_c,
            "last_p_leak":   round(twin.last_p_leak, 4),
            "is_healthy":    twin.is_healthy,
            "firmware_version": twin.firmware_version,
            "last_seen":     twin.last_seen.isoformat() if twin.last_seen else None,
        }, default=_json_serial))

    # Recent alerts (last 50)
    alerts = await store.alerts().list_all()
    for alert in alerts[:50]:
        await ws.send_text(json.dumps(_alert_to_msg(alert), default=_json_serial))

    # Recent work orders (last 20)
    wos = await store.work_orders().list_all()
    for wo in sorted(wos, key=lambda w: w.created_at, reverse=True)[:20]:
        await ws.send_text(json.dumps({
            "type":            "work_order",
            "work_order_id":   str(wo.work_order_id),
            "alert_id":        str(wo.alert_id),
            "status":          wo.status.value,
            "crew_id":         wo.crew_id,
            "eta_minutes":     wo.eta_minutes,
            "parts_required":  wo.parts_required,
            "notes":           wo.notes,
            "created_at":      wo.created_at.isoformat(),
            "completed_at":    wo.completed_at.isoformat() if wo.completed_at else None,
            "repair_cost_usd": wo.repair_cost_usd,
        }, default=_json_serial))

    # Hypotheses
    for h in store.hypotheses()[-20:]:
        await ws.send_text(json.dumps({
            "type":            "hypothesis",
            "hypothesis_id":   str(h.hypothesis_id),
            "lat":             h.lat,
            "lon":             h.lon,
            "uncertainty_m":   round(h.uncertainty_m, 1),
            "pipe_segment_id": h.pipe_segment_id,
            "confidence":      round(h.confidence, 4),
            "estimated_flow_lps": h.estimated_flow_lps,
            "created_at":      h.created_at.isoformat(),
            "n_sensors":       len(h.contributing_detection_ids),
        }, default=_json_serial))

    # Audit heartbeat
    ok, _ = audit_log.verify_chain()
    await ws.send_text(json.dumps({
        "type":         "heartbeat",
        "ts":           datetime.now(UTC).isoformat(),
        "audit_ok":     ok,
        "audit_events": len(audit_log.CHAIN),
        "ws_clients":   hub.count,
    }))


# ─────────────────────────── Entry point ─────────────────────────────────────

def main() -> None:
    uvicorn.run(
        "omni.ops_console.ws_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        # Reload only in development — comment out for production
        # reload=True,
    )


if __name__ == "__main__":
    main()
