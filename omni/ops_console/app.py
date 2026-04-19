"""Omni-Sense Ops Console.

A realtime Streamlit dashboard that stands up the full platform in-process
and streams live state: sensor health, detection firehose, triangulated
hypotheses on a map, alerts with FSM, work orders, audit chain integrity.

  streamlit run omni/ops_console/app.py
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

import pandas as pd
import streamlit as st

from omni.alerts import engine
from omni.audit import log as audit_log
from omni.cmms import service as cmms_service
from omni.common import store
from omni.common.bus import get_bus
from omni.common.store import SensorTwin
from omni.dispatch import router
from omni.edge.simulator import run_sensor
from omni.eep import orchestrator
from omni.notify import service as notify_service
from omni.spatial import fusion

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Omni-Sense · Ops Console", page_icon="💧", layout="wide")

# ─────────────── Global background loop (runs once) ───────────────────
if "loop_started" not in st.session_state:
    st.session_state["loop_started"] = False


def _start_platform() -> None:
    """Spin the bus + sensors in a background thread so Streamlit stays reactive."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    orchestrator.wire()
    fusion.wire()
    engine.wire()
    router.wire()
    cmms_service.wire()
    notify_service.wire()
    audit_log.wire()

    async def seed():
        sensors = [
            ("S-HAMRA-001", "beirut/hamra", 33.8978, 35.4828),
            ("S-HAMRA-002", "beirut/hamra", 33.8985, 35.4845),
            ("S-VERDUN-001", "beirut/verdun", 33.8840, 35.4910),
            ("S-ACHRAFIEH-001", "beirut/achrafieh", 33.8890, 35.5230),
        ]
        for sid, site, lat, lon in sensors:
            await store.twins().upsert_twin(
                SensorTwin(sensor_id=sid, site_id=site, lat=lat, lon=lon)
            )

        # Continuous scenario loop: quiet, occasional leak bursts
        schedule = [
            (8.0, "quiet"),
            (12.0, "leak"),
            (6.0, "pump"),
            (8.0, "quiet"),
        ]
        # run forever, rotating through schedules
        async def sensor_loop(sid, site, lat, lon, leaky: bool):
            while True:
                s = schedule if leaky else [(20.0, "quiet"), (8.0, "pump")]
                await run_sensor(sid, site, lat, lon, s, cadence_s=1.0)

        tasks = [
            asyncio.create_task(sensor_loop("S-HAMRA-001", "beirut/hamra", 33.8978, 35.4828, True)),
            asyncio.create_task(sensor_loop("S-HAMRA-002", "beirut/hamra", 33.8985, 35.4845, True)),
            asyncio.create_task(sensor_loop("S-VERDUN-001", "beirut/verdun", 33.8840, 35.4910, False)),
            asyncio.create_task(sensor_loop("S-ACHRAFIEH-001", "beirut/achrafieh", 33.8890, 35.5230, False)),
        ]
        await asyncio.gather(*tasks)

    async def main():
        bus = get_bus()
        await seed()
        await bus.run()  # runs forever

    loop.run_until_complete(main())


if not st.session_state["loop_started"]:
    t = threading.Thread(target=_start_platform, daemon=True)
    t.start()
    st.session_state["loop_started"] = True
    # give the bus a moment to boot
    time.sleep(1.0)


# ─────────────── UI ───────────────────────────────────────────────────
st.title("💧 Omni-Sense · Acoustic Water Intelligence")
st.caption(
    "Live view of the Beirut sensor mesh. Edge → EEP v2 → Spatial Fusion → "
    "Alerts → Dispatch → CMMS, with tamper-evident audit chain."
)

colA, colB, colC, colD, colE = st.columns(5)

auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_s = st.sidebar.slider("Interval (s)", 1, 10, 2)
st.sidebar.markdown("---")
st.sidebar.markdown("**Scenario**")
st.sidebar.caption(
    "Hamra-001 and Hamra-002 cycle through quiet → leak → pump → quiet. "
    "Verdun and Achrafieh stay quiet. Watch the pipeline react in real time."
)


def _snapshot():
    twins = list(store.twins().twins.values())
    alerts = asyncio.run(store.alerts().list_all())
    wos = asyncio.run(store.work_orders().list_all())
    hyps = list(store.hypotheses())
    return twins, alerts, wos, hyps


twins, alerts, wos, hyps = _snapshot()

# KPIs
with colA:
    st.metric("Sensors online", sum(1 for t in twins if t.is_healthy), f"/{len(twins)}")
with colB:
    active = sum(1 for a in alerts if a.state.value in ("NEW", "ACKNOWLEDGED", "DISPATCHED", "ON_SITE"))
    st.metric("Active alerts", active)
with colC:
    crit = sum(1 for a in alerts if a.severity.value == "critical")
    st.metric("Critical (lifetime)", crit)
with colD:
    loss_lph = sum(a.estimated_loss_lph or 0 for a in alerts if a.state.value != "RESOLVED" and a.state.value != "VERIFIED")
    st.metric("Water-loss rate", f"{loss_lph:,.0f} L/h")
with colE:
    ok, _ = audit_log.verify_chain()
    st.metric("Audit chain", "✓ intact" if ok else "✗ BROKEN", f"{len(audit_log.CHAIN)} events")

# Map
st.subheader("🗺️  Sensor mesh & leak hypotheses")
map_rows = []
for t in twins:
    map_rows.append(
        dict(
            lat=t.lat,
            lon=t.lon,
            label=f"{t.sensor_id} · p_leak={t.last_p_leak:.2f} · batt={t.battery_pct:.0f}%",
            size=60,
            color="#22c55e" if t.last_p_leak < 0.3 else ("#f59e0b" if t.last_p_leak < 0.6 else "#ef4444"),
        )
    )
for h in hyps[-20:]:
    map_rows.append(
        dict(
            lat=h.lat,
            lon=h.lon,
            label=f"HYPOTHESIS pipe={h.pipe_segment_id} conf={h.confidence:.0%}",
            size=180,
            color="#8b5cf6",
        )
    )
map_df = pd.DataFrame(map_rows)
if not map_df.empty:
    st.map(map_df, latitude="lat", longitude="lon", size="size", color="color")

# Alerts
st.subheader("🚨  Alerts (most recent 25)")
rows = []
for a in alerts[:25]:
    rows.append(
        dict(
            alert_id=str(a.alert_id)[:8],
            severity=a.severity.value,
            state=a.state.value,
            pipe=a.pipe_segment_id,
            loss_lph=a.estimated_loss_lph,
            crew=a.assigned_crew_id or "—",
            sla_due=a.sla_due_at.strftime("%H:%M:%S") if a.sla_due_at else "—",
            created=a.created_at.strftime("%H:%M:%S"),
        )
    )
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No alerts yet — sensors warming up.")

# Work orders
c1, c2 = st.columns(2)
with c1:
    st.subheader("🔧  Work orders")
    wo_rows = [
        dict(
            wo=str(w.work_order_id)[:8],
            status=w.status.value,
            crew=w.crew_id,
            eta_min=w.eta_minutes,
            cost=w.repair_cost_usd,
            parts=", ".join(w.parts_required[:3]),
        )
        for w in wos[-20:]
    ]
    if wo_rows:
        st.dataframe(pd.DataFrame(wo_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No work orders yet.")

with c2:
    st.subheader("📟  Notification inbox")
    nrows = [
        dict(
            sent_at=n.get("sent_at", "")[-8:],
            channel=n.get("channel"),
            severity=n.get("severity"),
            subject=n.get("subject"),
        )
        for n in list(notify_service.INBOX)[-20:][::-1]
    ]
    if nrows:
        st.dataframe(pd.DataFrame(nrows), use_container_width=True, hide_index=True)
    else:
        st.caption("No notifications yet.")

# Sensors & detection firehose
st.subheader("📡  Sensor twins")
t_rows = []
for t in twins:
    t_rows.append(
        dict(
            sensor=t.sensor_id,
            site=t.site_id,
            last_p_leak=round(t.last_p_leak, 3),
            battery=f"{t.battery_pct:.0f}%",
            temp_c=round(t.temperature_c, 1),
            healthy="✓" if t.is_healthy else "⚠",
            last_seen=t.last_seen.strftime("%H:%M:%S") if t.last_seen else "—",
        )
    )
st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

# Audit chain preview
with st.expander(f"🔐 Audit chain ({len(audit_log.CHAIN)} events, verify_key={audit_log._verify_key_hex[:16]}…)"):
    ev_rows = [
        dict(
            ts=e.ts.strftime("%H:%M:%S.%f")[:-3],
            actor=e.actor,
            action=e.action,
            resource=e.resource_type,
            hash=e.payload_hash_sha256[:12],
            sig=e.signature_ed25519[:12],
        )
        for e in audit_log.CHAIN[-30:][::-1]
    ]
    if ev_rows:
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

if auto_refresh:
    time.sleep(refresh_s)
    st.rerun()
