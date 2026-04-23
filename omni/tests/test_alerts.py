"""Alert engine: severity scoring, FSM transitions, SLA assignment."""
from uuid import uuid4

import pytest

from omni.alerts.engine import _score
from omni.common import store
from omni.common.schemas import (
    Alert,
    AlertState,
    LeakHypothesis,
    Severity,
)


def _make_hypothesis(pipe_id: str, confidence: float, flow: float) -> LeakHypothesis:
    return LeakHypothesis(
        contributing_detection_ids=[uuid4(), uuid4()],
        lat=33.8978,
        lon=35.4828,
        uncertainty_m=30.0,
        pipe_segment_id=pipe_id,
        estimated_flow_lps=flow,
        confidence=confidence,
    )


def test_critical_infrastructure_high_confidence_is_critical():
    h = _make_hypothesis("P-HAMRA-A12", confidence=0.95, flow=1.2)
    sev, score = _score(h)
    assert sev == Severity.CRITICAL
    assert score >= 80


def test_low_confidence_low_flow_not_critical():
    h = _make_hypothesis("P-VERDUN-B07", confidence=0.35, flow=0.1)
    sev, score = _score(h)
    assert sev in (Severity.INFO, Severity.LOW, Severity.MEDIUM)
    assert score < 60


def test_score_increases_with_confidence():
    h_low = _make_hypothesis("P-VERDUN-B07", confidence=0.3, flow=0.2)
    h_high = _make_hypothesis("P-VERDUN-B07", confidence=0.9, flow=0.2)
    _, s_low = _score(h_low)
    _, s_high = _score(h_high)
    assert s_high > s_low


def test_score_increases_with_flow():
    h_low = _make_hypothesis("P-VERDUN-B07", confidence=0.7, flow=0.1)
    h_high = _make_hypothesis("P-VERDUN-B07", confidence=0.7, flow=2.0)
    _, s_low = _score(h_low)
    _, s_high = _score(h_high)
    assert s_high > s_low


@pytest.mark.asyncio
async def test_alert_fsm_transition():
    # Create a fresh alert store for isolation
    a_store = store.AlertStore()
    alert = Alert(
        hypothesis_id=uuid4(),
        severity=Severity.HIGH,
        severity_score=65.0,
        title="Test alert",
        summary="test",
        lat=33.89,
        lon=35.48,
        sla_due_at=None,
    )
    await a_store.put(alert)

    # Transition to ACKNOWLEDGED
    updated = await a_store.transition(alert.alert_id, AlertState.ACKNOWLEDGED, "ack by operator")
    assert updated.state == AlertState.ACKNOWLEDGED
    assert len(updated.history) >= 1
    assert updated.history[-1]["to"] == "ACKNOWLEDGED"

    # Transition to DISPATCHED
    dispatched = await a_store.transition(alert.alert_id, AlertState.DISPATCHED, "crew assigned")
    assert dispatched.state == AlertState.DISPATCHED


@pytest.mark.asyncio
async def test_alert_store_lists_by_recency():
    a_store = store.AlertStore()
    for i in range(5):
        a = Alert(
            hypothesis_id=uuid4(),
            severity=Severity.LOW,
            severity_score=10.0,
            title=f"alert {i}",
            summary="",
            lat=33.0,
            lon=35.0,
        )
        await a_store.put(a)
    all_alerts = await a_store.list_all()
    assert len(all_alerts) == 5
    # Most recent first
    assert all_alerts[0].created_at >= all_alerts[-1].created_at
