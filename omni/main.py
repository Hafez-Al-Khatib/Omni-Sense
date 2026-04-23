"""Omni-Sense Platform v2 — single-process runner.

Wires every service into the in-memory bus and runs a realistic scenario:
three sensors across Beirut, quiet for a bit, then two of them pick up a
leak signature at the same time, EEP v2 fires, spatial fusion triangulates,
alert engine scores, dispatch routes a crew, CMMS completes the repair,
audit log captures everything.

  python -m omni.main                # run demo scenario, exit when stable
  python -m omni.main --forever      # keep running for manual inspection
  python -m omni.main --use-gateway  # route frames through MQTT gateway path
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys

from omni.alerts import engine
from omni.audit import log as audit_log
from omni.cmms import service as cmms_service
from omni.common import store
from omni.common.bus import get_bus
from omni.common.store import SensorTwin
from omni.common.tracing import configure_tracing
from omni.dispatch import router
from omni.edge.gateway import StubMQTTGateway
from omni.edge.simulator import run_sensor
from omni.eep import orchestrator
from omni.mlops import feedback_watcher
from omni.mlops.drift_detector import get_detector
from omni.mlops.retraining_trigger import get_trigger
from omni.notify import service as notify_service
from omni.spatial import fusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)-10s │ %(message)s",
    datefmt="%H:%M:%S",
)


SENSORS = [
    dict(sensor_id="S-HAMRA-001", site_id="beirut/hamra", lat=33.8978, lon=35.4828),
    dict(sensor_id="S-HAMRA-002", site_id="beirut/hamra", lat=33.8985, lon=35.4845),
    dict(sensor_id="S-VERDUN-001", site_id="beirut/verdun", lat=33.8840, lon=35.4910),
]

# Scenario: 5 s quiet, 15 s leak (Hamra sensors), 5 s cool-down
LEAK_SCHEDULE = [(5.0, "quiet"), (15.0, "leak"), (5.0, "quiet")]
QUIET_SCHEDULE = [(25.0, "quiet")]


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


def wire_everything() -> None:
    configure_tracing("omni-platform")
    orchestrator.wire()
    fusion.wire()
    engine.wire()
    router.wire()
    cmms_service.wire()
    notify_service.wire()
    audit_log.wire()
    # MLOps flywheel: drift detector + retraining trigger
    get_detector().wire()
    get_trigger().wire()


async def _run_via_simulator() -> None:
    """Direct bus ingestion via the edge simulator (default, fastest)."""
    tasks = [
        asyncio.create_task(
            run_sensor(
                sensor_id=SENSORS[0]["sensor_id"],
                site_id=SENSORS[0]["site_id"],
                lat=SENSORS[0]["lat"],
                lon=SENSORS[0]["lon"],
                regime_schedule=LEAK_SCHEDULE,
                cadence_s=0.5,
            )
        ),
        asyncio.create_task(
            run_sensor(
                sensor_id=SENSORS[1]["sensor_id"],
                site_id=SENSORS[1]["site_id"],
                lat=SENSORS[1]["lat"],
                lon=SENSORS[1]["lon"],
                regime_schedule=LEAK_SCHEDULE,
                cadence_s=0.5,
            )
        ),
        asyncio.create_task(
            run_sensor(
                sensor_id=SENSORS[2]["sensor_id"],
                site_id=SENSORS[2]["site_id"],
                lat=SENSORS[2]["lat"],
                lon=SENSORS[2]["lon"],
                regime_schedule=QUIET_SCHEDULE,
                cadence_s=0.5,
            )
        ),
    ]
    await asyncio.gather(*tasks)


async def _run_via_gateway() -> None:
    """Route frames through StubMQTTGateway, exercising full payload validation.

    Same scenario as the simulator, but every acoustic frame is serialized to
    JSON, validated for size, parsed, schema-checked, and forwarded via the
    same code path that production MQTT messages travel.
    """
    log = logging.getLogger("main")
    log.info(
        "MQTT stub gateway mode — frames validated via omni/edge/gateway.py"
    )

    # Map each sensor to its regime schedule
    sensor_configs = [
        (SENSORS[0], LEAK_SCHEDULE),
        (SENSORS[1], LEAK_SCHEDULE),
        (SENSORS[2], QUIET_SCHEDULE),
    ]
    sensors  = [s for s, _ in sensor_configs]
    schedules = [sched for _, sched in sensor_configs]

    gw = StubMQTTGateway(sensors, cadence_s=0.5)
    await gw.run(schedules)


async def run_demo(forever: bool = False, use_gateway: bool = False) -> None:
    wire_everything()
    await _seed_twins()
    bus = get_bus()
    bus_task = asyncio.create_task(bus.run(), name="bus")
    # Start feedback watcher as a background task — closes the active-learning loop
    feedback_watcher.start()

    # Choose ingestion path
    if use_gateway:
        await _run_via_gateway()
    else:
        await _run_via_simulator()

    # Give the downstream pipeline a moment to drain
    await asyncio.sleep(2.0)

    if not forever:
        bus.stop()
        bus_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bus_task

    # Final report
    all_alerts = await store.alerts().list_all()
    all_wos = await store.work_orders().list_all()
    ok, bad = audit_log.verify_chain()

    drift_report = get_detector().latest_report
    rul_schedule = cmms_service.get_maintenance_schedule()

    print("\n" + "═" * 78)
    print("  OMNI-SENSE v2 · END-OF-RUN REPORT")
    print("═" * 78)
    print(f"  Sensors online      : {len(store.twins().twins)}")
    print(f"  Frames processed    : {sum(len(dq) for dq in store.twins().detection_window.values())}")
    print(f"  Hypotheses raised   : {len(store.hypotheses())}")
    print(f"  Alerts              : {len(all_alerts)}")
    print(f"  Work orders closed  : {sum(1 for w in all_wos if w.completed_at is not None)}/{len(all_wos)}")
    print(f"  Audit chain         : {'✓ VALID' if ok else f'✗ BROKEN at {bad}'} ({len(audit_log.CHAIN)} events)")
    print(f"  Notifications sent  : {len(notify_service.INBOX)}")
    if drift_report:
        print(f"  Drift level         : {drift_report.drift_level.upper()} (PSI_max={drift_report.psi_max:.3f})")
    print("-" * 78)
    for a in all_alerts[:5]:
        print(f"  [{a.severity.value:>8}] {a.state.value:>10} · {a.title}")
        print(f"             pipe={a.pipe_segment_id} crew={a.assigned_crew_id} loss={a.estimated_loss_lph} L/h")
    if rul_schedule:
        print("-" * 78)
        print("  PREDICTIVE MAINTENANCE SCHEDULE (by urgency):")
        for r in rul_schedule[:4]:
            print(f"  [{r.risk_tier:>8}] {r.segment_id:<22} RUL={r.rul_days:>6.0f}d  P(30d)={r.survival_30d:.3f}")
    print("═" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="Omni-Sense Platform v2 demo runner")
    ap.add_argument("--forever", action="store_true",
                    help="Keep the bus running after the scenario finishes")
    ap.add_argument("--use-gateway", action="store_true",
                    help="Route frames via StubMQTTGateway (tests full validation path)")
    args = ap.parse_args()
    try:
        asyncio.run(run_demo(forever=args.forever, use_gateway=args.use_gateway))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
