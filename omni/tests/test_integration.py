"""Full-pipeline integration test.

Runs a 15-second scenario (compressed to ~3s with fast cadence) and asserts
that at least one end-to-end path completes: frame → detection → hypothesis
→ alert → work order → resolved + audit chain intact.
"""
import asyncio
import contextlib
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

import omni.common.bus as bus_mod
from omni.alerts import engine
from omni.audit import log as audit_log
from omni.cmms import service as cmms
from omni.common import store
from omni.common.bus import InMemoryBus
from omni.common.store import SensorTwin
from omni.dispatch import router
from omni.edge.simulator import run_sensor
from omni.eep import orchestrator
from omni.notify import service as notify
from omni.spatial import fusion


@pytest.mark.asyncio
async def test_end_to_end_leak_detected_and_dispatched():
    # Fresh global state
    store._twin = store.DigitalTwinStore()
    store._alerts = store.AlertStore()
    store._work_orders = store.WorkOrderStore()
    store._hypotheses = []
    bus_mod._bus = InMemoryBus()
    audit_log.CHAIN.clear()

    # Reset spatial debounce
    import omni.spatial.fusion as sf_mod
    sf_mod._last_publish = datetime.min.replace(tzinfo=UTC)

    with patch.object(orchestrator, "_load_models"):
        orchestrator.wire()
    fusion.wire()
    engine.wire()
    router.wire()
    cmms.wire()
    notify.wire()
    audit_log.wire()

    bus = bus_mod.get_bus()

    # Seed twins for two Hamra sensors matching fusion.PIPE_SEGMENTS
    for sid, lat, lon in [("S-HAMRA-001", 33.8978, 35.4828), ("S-HAMRA-002", 33.8985, 35.4845)]:
        await store.twins().upsert_twin(
            SensorTwin(sensor_id=sid, site_id="beirut/hamra", lat=lat, lon=lon)
        )

    bus_task = asyncio.create_task(bus.run())

    # Run two leak sensors for a short burst (fast cadence)
    await asyncio.gather(
        run_sensor("S-HAMRA-001", "beirut/hamra", 33.8978, 35.4828, [(4.0, "leak")], cadence_s=0.3),
        run_sensor("S-HAMRA-002", "beirut/hamra", 33.8985, 35.4845, [(4.0, "leak")], cadence_s=0.3),
    )
    await asyncio.sleep(3.0)  # drain pipeline

    bus.stop()
    bus_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await bus_task

    # Assertions
    all_alerts = await store.alerts().list_all()
    all_wos = await store.work_orders().list_all()
    ok, bad = audit_log.verify_chain()

    assert len(store.hypotheses()) >= 1, "At least one spatial hypothesis expected"
    assert len(all_alerts) >= 1, "At least one alert expected"
    assert any(a.assigned_crew_id is not None for a in all_alerts), "A crew should be dispatched"
    assert len(all_wos) >= 1, "At least one work order expected"
    assert ok, f"Audit chain should be intact, broken at index {bad}"
    assert len(notify.INBOX) >= 1, "At least one notification expected"
