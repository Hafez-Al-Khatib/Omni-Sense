"""CMMS — Computerized Maintenance Management System.

Tracks work orders through completion, stores the repair record, maintains a
per-segment pipe registry, and computes RUL after every repair so the ops
console can show a predictive maintenance schedule.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from omni.common.bus import Topics, get_bus
from omni.common.schemas import AlertState, WorkOrder, WorkOrderStatus
from omni.common import store
from omni.cmms.rul_model import RULPrediction, predict_rul

log = logging.getLogger("cmms")


# ─────────────────────────── Pipe segment registry ────────────────────────────

@dataclass
class PipeSegment:
    """Live state for one named pipe segment."""
    segment_id: str
    install_year: int = 2000
    material: str = "PVC"          # "PVC" | "Steel" | "Cast_Iron"
    topology: str = "Looped"       # "Looped" | "Branched"
    pressure_bar: float = 4.5
    repair_count: int = 0
    last_repair: Optional[datetime] = None
    last_rul: Optional[RULPrediction] = None
    leak_detections_30d: int = 0


_PIPE_REGISTRY: dict[str, PipeSegment] = {
    "P-HAMRA-A12": PipeSegment(
        "P-HAMRA-A12", install_year=1988, material="Cast_Iron",
        topology="Branched", pressure_bar=6.5, repair_count=3,
    ),
    "P-HAMRA-A13": PipeSegment(
        "P-HAMRA-A13", install_year=2005, material="Steel",
        topology="Branched", pressure_bar=5.2, repair_count=1,
    ),
    "P-VERDUN-B07": PipeSegment(
        "P-VERDUN-B07", install_year=2018, material="PVC",
        topology="Looped", pressure_bar=3.5, repair_count=0,
    ),
    "P-ACHRAFIEH-C02": PipeSegment(
        "P-ACHRAFIEH-C02", install_year=2003, material="Steel",
        topology="Looped", pressure_bar=5.0, repair_count=2,
    ),
}

_LATEST_RULS: list[RULPrediction] = []


def _refresh_rul(seg: PipeSegment, extra_detections: int = 0) -> RULPrediction:
    """Re-run the RUL model for a pipe segment and cache the result."""
    age = datetime.now(timezone.utc).year - seg.install_year
    detections = max(0, seg.leak_detections_30d + extra_detections)
    rul = predict_rul(
        segment_id=seg.segment_id,
        pipe_age_years=float(age),
        repair_count=seg.repair_count,
        pressure_bar=seg.pressure_bar,
        is_cast_iron=seg.material == "Cast_Iron",
        is_branched=seg.topology == "Branched",
        leak_detections_30d=detections,
    )
    seg.last_rul = rul
    # Keep global list up to date (replace old entry for this segment)
    global _LATEST_RULS
    _LATEST_RULS = [r for r in _LATEST_RULS if r.segment_id != seg.segment_id]
    _LATEST_RULS.append(rul)
    _LATEST_RULS.sort(key=lambda r: r.rul_days)   # most urgent first
    return rul


def get_maintenance_schedule() -> list[RULPrediction]:
    """Return all known RUL predictions sorted by urgency (lowest RUL first)."""
    return list(_LATEST_RULS)


def get_registry() -> dict[str, PipeSegment]:
    return _PIPE_REGISTRY


# ─────────────────────────── Event handlers ───────────────────────────────────

async def on_work_order(payload: dict) -> None:
    wo = WorkOrder(**payload)

    # Simulate successful repair
    cost = round(random.uniform(180, 1200), 2)
    notes = (
        "Repair completed. Clamp installed, pressure test passed. "
        "No secondary damage detected."
    )
    completed = await store.work_orders().complete(wo.work_order_id, cost, notes)
    await store.work_orders().put(completed)

    updated = await store.alerts().transition(
        wo.alert_id, AlertState.RESOLVED, note=f"WO {wo.work_order_id} closed"
    )
    await store.alerts().put(updated)
    await get_bus().publish(Topics.ALERT_STATE, updated)

    # Find the pipe segment this work order affects (via the alert)
    alert = await store.alerts().get(wo.alert_id)
    pipe_id = alert.pipe_segment_id if alert else None

    if pipe_id and pipe_id in _PIPE_REGISTRY:
        seg = _PIPE_REGISTRY[pipe_id]
        seg.repair_count += 1
        seg.last_repair = datetime.now(timezone.utc)
        rul = _refresh_rul(seg)
        completed.mtbf_days = rul.rul_days
        await store.work_orders().put(completed)
        log.info(
            "WO %s closed  pipe=%s  repair_count=%d  RUL=%.0fd [%s]  cost=$%.2f",
            wo.work_order_id, pipe_id, seg.repair_count,
            rul.rul_days, rul.risk_tier, cost,
        )
    else:
        completed.mtbf_days = round(random.uniform(180, 900), 0)
        await store.work_orders().put(completed)
        log.info("WO %s closed  cost=$%.2f", wo.work_order_id, cost)


async def on_detection(payload: dict) -> None:
    """Track detection counts per pipe segment for the RUL model."""
    from omni.common.schemas import DetectionResult
    det = DetectionResult(**payload)
    if not det.is_leak:
        return
    # Map sensor → pipe segment via the hypothesis store (best-effort)
    for hyp in store.hypotheses():
        if det.detection_id in hyp.contributing_detection_ids:
            pid = hyp.pipe_segment_id
            if pid and pid in _PIPE_REGISTRY:
                _PIPE_REGISTRY[pid].leak_detections_30d += 1
                # Re-score RUL immediately — more detections = more urgent
                _refresh_rul(_PIPE_REGISTRY[pid])
            break


def bootstrap_rul() -> None:
    """Compute initial RUL for all known segments at startup."""
    for seg in _PIPE_REGISTRY.values():
        try:
            _refresh_rul(seg)
        except Exception:
            log.exception("RUL bootstrap failed for %s", seg.segment_id)
    log.info(
        "RUL bootstrap complete — %d segments evaluated",
        len(_PIPE_REGISTRY),
    )
    for rul in _LATEST_RULS:
        log.info(
            "  %s  RUL=%.0fd  [%s]  P(30d)=%.2f",
            rul.segment_id, rul.rul_days, rul.risk_tier, rul.survival_30d,
        )


def wire() -> None:
    get_bus().subscribe(Topics.WORK_ORDER, on_work_order)
    get_bus().subscribe(Topics.DETECTION, on_detection)
    bootstrap_rul()
