"""Dispatch & Routing.

Accepts new alerts, picks the best available crew, builds a work order,
pushes notifications, and transitions the alert FSM.

Crew selection here is a nearest-available greedy pick over a static roster.
Production: OR-Tools VRP solver with skills/parts/shift constraints.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from omni.common.bus import Topics, get_bus
from omni.common.schemas import (
    Alert,
    AlertState,
    Severity,
    WorkOrder,
    WorkOrderStatus,
)
from omni.common import store

log = logging.getLogger("dispatch")

# Static crew roster for demo. Each crew has a base station + skill set.
CREWS = [
    {
        "id": "CREW-01",
        "name": "Hamra Rapid Response",
        "lat": 33.8975,
        "lon": 35.4800,
        "skills": {"leak_repair", "pressure_test", "excavation"},
        "on_shift": True,
        "current_load": 0,
    },
    {
        "id": "CREW-02",
        "name": "Verdun North",
        "lat": 33.8825,
        "lon": 35.4895,
        "skills": {"leak_repair", "welding"},
        "on_shift": True,
        "current_load": 0,
    },
    {
        "id": "CREW-03",
        "name": "Achrafieh East",
        "lat": 33.8880,
        "lon": 35.5240,
        "skills": {"leak_repair", "large_bore", "excavation"},
        "on_shift": True,
        "current_load": 1,
    },
]

AVG_TRAVEL_KPH = 28  # Beirut traffic reality check


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _choose_crew(alert: Alert) -> Optional[dict]:
    needed = {"leak_repair"}
    if alert.severity in (Severity.HIGH, Severity.CRITICAL):
        needed.add("excavation")
    candidates = [c for c in CREWS if c["on_shift"] and needed.issubset(c["skills"])]
    if not candidates:
        return None
    scored = []
    for c in candidates:
        d_km = _haversine_km(alert.lat, alert.lon, c["lat"], c["lon"])
        eta_min = (d_km / AVG_TRAVEL_KPH) * 60
        # penalize loaded crews so we spread work
        score = eta_min + 15 * c["current_load"]
        scored.append((score, eta_min, d_km, c))
    scored.sort()
    _, eta, dist, crew = scored[0]
    crew["_eta_min"] = max(1, int(eta))  # always at least 1 min (mobilise + travel)
    crew["_distance_km"] = round(dist, 2)
    return crew


async def on_alert_new(payload: dict) -> None:
    alert = Alert(**payload)
    crew = _choose_crew(alert)
    if crew is None:
        log.warning("no crew available for alert %s", alert.alert_id)
        return
    parts = ["pipe_clamp_50mm", "gasket_seal", "repair_sleeve"]
    if alert.severity == Severity.CRITICAL:
        parts.append("emergency_bypass_kit")

    wo = WorkOrder(
        alert_id=alert.alert_id,
        status=WorkOrderStatus.DISPATCHED,
        crew_id=crew["id"],
        eta_minutes=crew["_eta_min"],
        parts_required=parts,
        notes=(
            f"Auto-dispatched by Omni-Sense. "
            f"Distance {crew['_distance_km']} km, ETA {crew['_eta_min']} min."
        ),
    )
    crew["current_load"] += 1
    await store.work_orders().put(wo)
    updated = await store.alerts().transition(
        alert.alert_id,
        AlertState.DISPATCHED,
        note=f"assigned {crew['id']} ({crew['name']})",
    )
    updated.assigned_crew_id = crew["id"]
    await store.alerts().put(updated)

    await get_bus().publish(Topics.WORK_ORDER, wo)
    await get_bus().publish(Topics.ALERT_STATE, updated)

    # Fire notifications
    await get_bus().publish(
        Topics.NOTIFY,
        {
            "channel": "sms",
            "to": crew["name"],
            "severity": alert.severity.value,
            "subject": f"[{alert.severity.value.upper()}] Leak dispatch {alert.title}",
            "body": (
                f"Head to ({alert.lat:.5f},{alert.lon:.5f}) — pipe "
                f"{alert.pipe_segment_id}. ETA {crew['_eta_min']} min. "
                f"Parts: {', '.join(parts)}. WO {wo.work_order_id}."
            ),
        },
    )
    log.info(
        "dispatched %s to %s ETA=%dm severity=%s",
        crew["id"], alert.alert_id, crew["_eta_min"], alert.severity.value,
    )


def wire() -> None:
    get_bus().subscribe(Topics.ALERT_NEW, on_alert_new)
