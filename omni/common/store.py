"""Persistent stores — in-memory for the demo, swap for Postgres/Redis in prod.

Everything here is intentionally small: the goal is to define the *interfaces*
the services use, so that Miriam can later swap in Timescale + Redis + PostGIS
without touching business logic.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from .schemas import (
    Alert,
    AlertState,
    DetectionResult,
    LeakHypothesis,
    WorkOrder,
    WorkOrderStatus,
)

log = logging.getLogger("store")


@dataclass
class SensorTwin:
    sensor_id: str
    site_id: str
    lat: float
    lon: float
    last_seen: datetime | None = None
    battery_pct: float = 100.0
    temperature_c: float = 25.0
    firmware_version: str = "unknown"
    rolling_noise_floor_db: float = -60.0
    last_p_leak: float = 0.0
    is_healthy: bool = True


@dataclass
class DigitalTwinStore:
    """RedisTimeSeries stand-in. Per-sensor live state + bounded rolling window."""

    twins: dict[str, SensorTwin] = field(default_factory=dict)
    detection_window: dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=256))
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def upsert_twin(self, twin: SensorTwin) -> None:
        async with self._lock:
            self.twins[twin.sensor_id] = twin

    async def update_telemetry(
        self, sensor_id: str, battery: float, temp: float, fw: str
    ) -> None:
        async with self._lock:
            t = self.twins.setdefault(
                sensor_id,
                SensorTwin(sensor_id=sensor_id, site_id="unknown", lat=0.0, lon=0.0),
            )
            t.last_seen = datetime.now(UTC)
            t.battery_pct = battery
            t.temperature_c = temp
            t.firmware_version = fw
            t.is_healthy = battery > 15 and temp < 70

    async def record_detection(self, det: DetectionResult) -> None:
        async with self._lock:
            self.detection_window[det.sensor_id].append(det)
            if det.sensor_id in self.twins:
                self.twins[det.sensor_id].last_p_leak = det.fused_p_leak

    async def recent_detections(self, sensor_id: str, n: int = 32) -> list[DetectionResult]:
        async with self._lock:
            return list(self.detection_window[sensor_id])[-n:]

    async def all_recent_leaks(self, min_p: float = 0.7, horizon_s: float = 30.0) -> list[DetectionResult]:
        now = datetime.now(UTC)
        out = []
        async with self._lock:
            for dq in self.detection_window.values():
                for det in dq:
                    if det.fused_p_leak >= min_p and (now - det.captured_at).total_seconds() <= horizon_s:
                        out.append(det)
        return out


@dataclass
class AlertStore:
    by_id: dict[UUID, Alert] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def put(self, alert: Alert) -> None:
        async with self._lock:
            self.by_id[alert.alert_id] = alert

    async def get(self, alert_id: UUID) -> Alert | None:
        async with self._lock:
            return self.by_id.get(alert_id)

    async def transition(self, alert_id: UUID, new_state: AlertState, note: str = "") -> Alert:
        async with self._lock:
            alert = self.by_id[alert_id]
            alert.history.append(
                {
                    "from": alert.state.value,
                    "to": new_state.value,
                    "at": datetime.now(UTC).isoformat(),
                    "note": note,
                }
            )
            alert.state = new_state
            alert.updated_at = datetime.now(UTC)
            return alert

    async def list_all(self) -> list[Alert]:
        async with self._lock:
            return sorted(self.by_id.values(), key=lambda a: a.created_at, reverse=True)


@dataclass
class WorkOrderStore:
    by_id: dict[UUID, WorkOrder] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def put(self, wo: WorkOrder) -> None:
        async with self._lock:
            self.by_id[wo.work_order_id] = wo

    async def complete(self, wo_id: UUID, cost: float, notes: str) -> WorkOrder:
        async with self._lock:
            wo = self.by_id[wo_id]
            wo.status = WorkOrderStatus.COMPLETED
            wo.completed_at = datetime.now(UTC)
            wo.repair_cost_usd = cost
            wo.notes = notes
            return wo

    async def list_all(self) -> list[WorkOrder]:
        async with self._lock:
            return list(self.by_id.values())


# Module-level singletons for the demo
_twin = DigitalTwinStore()
_alerts = AlertStore()
_work_orders = WorkOrderStore()
_hypotheses: list[LeakHypothesis] = []


def twins() -> DigitalTwinStore:
    return _twin


def alerts() -> AlertStore:
    return _alerts


def work_orders() -> WorkOrderStore:
    return _work_orders


def hypotheses() -> list[LeakHypothesis]:
    return _hypotheses
