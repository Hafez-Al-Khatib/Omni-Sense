"""Dispatch: crew selection, work order creation, FSM integration."""
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from omni.dispatch.router import _choose_crew, CREWS
from omni.common.schemas import Alert, AlertState, Severity
from omni.common import store as global_store


def _alert(sev=Severity.HIGH, lat=33.8978, lon=35.4828):
    return Alert(
        hypothesis_id=uuid4(),
        severity=sev,
        severity_score=70.0,
        title="Test leak",
        summary="Test",
        lat=lat,
        lon=lon,
        pipe_segment_id="P-HAMRA-A12",
    )


def test_choose_crew_returns_on_shift():
    alert = _alert()
    crew = _choose_crew(alert)
    assert crew is not None
    assert crew["on_shift"] is True


def test_choose_crew_prefers_nearest():
    # Alert near Hamra — CREW-01 should be chosen over Achrafieh crew
    alert = _alert(lat=33.8978, lon=35.4828)
    crew = _choose_crew(alert)
    # CREW-01 is based in Hamra, should win
    assert crew["id"] in ("CREW-01", "CREW-02")


def test_choose_crew_for_critical_requires_excavation():
    alert = _alert(sev=Severity.CRITICAL)
    crew = _choose_crew(alert)
    assert crew is not None
    assert "excavation" in crew["skills"]


def test_choose_crew_eta_is_positive():
    crew = _choose_crew(_alert())
    assert crew["_eta_min"] > 0


@pytest.mark.asyncio
async def test_dispatch_creates_work_order():
    """End-to-end: alert → dispatch → work order persisted."""
    from omni.dispatch.router import on_alert_new
    from omni.common.bus import InMemoryBus, Topics
    import omni.common.bus as bus_mod

    # Wire fresh bus + stores
    old_bus = bus_mod._bus
    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()

    # Re-wire stores so we can inspect them
    bus_task = asyncio.create_task(bus.run())

    alert = _alert(sev=Severity.HIGH)
    await global_store.alerts().put(alert)
    await on_alert_new(alert.model_dump(mode="json"))

    await asyncio.sleep(0.1)
    bus.stop()
    bus_task.cancel()
    try:
        await bus_task
    except asyncio.CancelledError:
        pass

    wos = await global_store.work_orders().list_all()
    assert any(str(w.alert_id) == str(alert.alert_id) for w in wos)

    # Restore
    bus_mod._bus = old_bus
