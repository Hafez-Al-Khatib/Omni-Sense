"""Alert Engine & Severity Scoring.

Turns spatial hypotheses into Alerts, assigns severity, manages the FSM
(NEW→ACK→DISPATCHED→ON_SITE→RESOLVED→VERIFIED), applies SLA timers, and
emits downstream events.

Severity blends: hypothesis confidence, estimated flow rate, pipe criticality,
population density around the leak, and whether the pipe serves a hospital
or school. In the demo we keep the blend simple but the weights are explicit
so ops can tune them without touching code.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from omni.common.bus import Topics, get_bus
from omni.common.schemas import (
    Alert,
    AlertState,
    LeakHypothesis,
    Severity,
)
from omni.common import store

log = logging.getLogger("alerts")

# Pipe criticality table — in prod this is a PostGIS attribute
PIPE_CRITICALITY = {
    "P-HAMRA-A12": {"critical_infra": True, "population_density": "high"},
    "P-HAMRA-A13": {"critical_infra": False, "population_density": "high"},
    "P-VERDUN-B07": {"critical_infra": False, "population_density": "medium"},
    "P-ACHRAFIEH-C02": {"critical_infra": True, "population_density": "high"},
}

# SLA table by severity (minutes to acknowledge)
SLA_MINUTES = {
    Severity.CRITICAL: 5,
    Severity.HIGH: 30,
    Severity.MEDIUM: 120,
    Severity.LOW: 480,
    Severity.INFO: 1440,
}


def _score(h: LeakHypothesis) -> tuple[Severity, float]:
    flow = h.estimated_flow_lps or 0.0
    conf = h.confidence
    meta = PIPE_CRITICALITY.get(h.pipe_segment_id or "", {})

    score = 0.0
    score += 40 * conf
    score += min(25, flow * 15)  # l/s → 25 max
    if meta.get("critical_infra"):
        score += 20
    pop = meta.get("population_density", "low")
    score += {"high": 15, "medium": 8, "low": 2}.get(pop, 0)

    if score >= 80:
        sev = Severity.CRITICAL
    elif score >= 60:
        sev = Severity.HIGH
    elif score >= 40:
        sev = Severity.MEDIUM
    elif score >= 20:
        sev = Severity.LOW
    else:
        sev = Severity.INFO
    return sev, min(100.0, score)


async def on_hypothesis(payload: dict) -> None:
    h = LeakHypothesis(**payload)
    sev, s = _score(h)
    sla = datetime.now(timezone.utc) + timedelta(minutes=SLA_MINUTES[sev])
    alert = Alert(
        hypothesis_id=h.hypothesis_id,
        severity=sev,
        severity_score=s,
        title=f"Suspected leak on {h.pipe_segment_id or 'unknown segment'}",
        summary=(
            f"Triangulated from {len(h.contributing_detection_ids)} sensors at "
            f"({h.lat:.5f}, {h.lon:.5f}) ±{h.uncertainty_m:.0f}m. "
            f"Est. flow {h.estimated_flow_lps} L/s. Confidence {h.confidence:.0%}."
        ),
        lat=h.lat,
        lon=h.lon,
        pipe_segment_id=h.pipe_segment_id,
        estimated_loss_lph=round((h.estimated_flow_lps or 0) * 3600, 1),
        sla_due_at=sla,
    )
    alert.history.append({
        "at": datetime.now(timezone.utc).isoformat(),
        "note": f"created severity={sev.value} score={s:.1f}",
    })
    await store.alerts().put(alert)
    await get_bus().publish(Topics.ALERT_NEW, alert)
    log.info("alert %s severity=%s score=%.1f", alert.alert_id, sev.value, s)


def wire() -> None:
    get_bus().subscribe(Topics.HYPOTHESIS, on_hypothesis)
