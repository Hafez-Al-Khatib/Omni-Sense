# Omni-Sense Platform v2 — Industry-Grade Upgrade

**Acoustic water-infrastructure diagnostics for Lebanese urban networks.**  
AUB Capstone EECE503N/EECE798N · Spring 2026 · Hafez · Miriam · Reem

---

## What was added

The original project had a working ML inference stack (IEP1–4 + EEP) exposed over HTTP. This `omni/` package adds the full production intelligence layer on top:

| Layer | Service | File |
|---|---|---|
| **Edge** | Sensor simulator (VAD + gain-norm + PCM16 frames) | `edge/simulator.py` |
| **Ingestion** | In-memory event bus (Kafka-compatible interface) | `common/bus.py` |
| **Intelligence** | EEP v2 — async fan-out to 5 ML heads, fusion, OOD gate | `eep/orchestrator.py` |
| **Intelligence** | Spatial fusion + triangulation + pipe snapping | `spatial/fusion.py` |
| **Intelligence** | Digital twin per-sensor state store | `common/store.py` |
| **Action** | Alert engine FSM + severity scoring | `alerts/engine.py` |
| **Action** | Dispatch & routing (nearest-crew greedy, OR-Tools ready) | `dispatch/router.py` |
| **Action** | CMMS — work order lifecycle + cost + MTBF | `cmms/service.py` |
| **Action** | Multi-channel notifications (SMS/FCM/email stubs) | `notify/service.py` |
| **Compliance** | WORM audit log (Ed25519 + Merkle hash chain) | `audit/log.py` |
| **UI** | Real-time Streamlit ops console | `ops_console/app.py` |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r omni/requirements.txt

# 2. Run the demo scenario (exits automatically)
python -m omni.main

# 3. Run the live ops console
streamlit run omni/ops_console/app.py

# 4. Run all tests
pytest omni/tests/ -v
```

---

## Architecture

```
 Acoustic sensors (edge)
        │  AcousticFrame (PCM16 + SNR + VAD)
        ▼
 ┌─────────────────────────────────────┐
 │          In-Memory Bus              │  → Redpanda/Kafka in prod
 └──────────────┬──────────────────────┘
                │
      ┌─────────▼──────────┐
      │    EEP v2           │  Fan-out with hard timeout budgets
      │  ┌──── XGBoost ───┐ │  XGB 30ms · RF 30ms · CNN 150ms
      │  ├──── RF ─────── │ │  IF 20ms  · OOD 40ms
      │  ├──── CNN ──────┤ │  Fusion: 0.45·XGB + 0.25·RF +
      │  ├──── Iso.Forest│ │           0.25·CNN + 0.05·IF
      │  └──── OOD Gate ─┘ │  MC-Dropout uncertainty estimate
      └─────────┬───────────┘  SHAP top-3 feature attributions
                │ DetectionResult
      ┌─────────▼───────────┐
      │  Spatial Fusion      │  Kalman centroid · PostGIS snap
      │  + Triangulation     │  Correlation window: 12 s
      └─────────┬────────────┘  Min 2 sensors for hypothesis
                │ LeakHypothesis (lat/lon ±m, pipe_id, flow L/s)
      ┌─────────▼────────────┐
      │   Alert Engine       │  Severity = f(confidence, flow,
      │   FSM scorer         │    pipe criticality, population)
      └──────┬───────────────┘  SLA timers: CRIT=5min, HIGH=30min
             │ Alert (NEW)
      ┌──────▼──────────────┐
      │  Dispatch & Routing  │  Nearest-available crew
      │  + CMMS              │  Work order lifecycle
      └──────┬───────────────┘  Cost + MTBF logging
             │
      ┌──────▼──────────────┐
      │  Notifications       │  SMS · Push · Email (stubs)
      └─────────────────────┘

 All events ──► WORM Audit Log (Ed25519 + Merkle chain)
```

---

## Alert FSM

```
NEW ──► ACKNOWLEDGED ──► DISPATCHED ──► ON_SITE ──► RESOLVED ──► VERIFIED
 │                                                                    │
 └──────────────────────────────────────────────────────────────────►┘
         (also: SUPPRESSED, FALSE_POSITIVE)
```

---

## Severity scoring

| Factor | Weight |
|---|---|
| Hypothesis confidence (0–1) | × 40 |
| Estimated flow L/s (cap 25) | × 15 |
| Critical infrastructure (hospital/school) | +20 |
| Population density high/medium/low | +15 / +8 / +2 |
| **≥ 80 → CRITICAL · ≥ 60 → HIGH · ≥ 40 → MEDIUM** | |

---

## SLA table

| Severity | Acknowledge within |
|---|---|
| CRITICAL | 5 minutes |
| HIGH | 30 minutes |
| MEDIUM | 2 hours |
| LOW | 8 hours |
| INFO | 24 hours |

---

## Ownership

| Owner | Services |
|---|---|
| **Hafez** (ML/AI) | `eep/`, `spatial/`, `common/schemas.py`, `audit/` |
| **Miriam** (DevOps) | `common/bus.py`, `common/store.py`, `requirements.txt`, docker-compose |
| **Reem** (Frontend) | `ops_console/`, `notify/`, `alerts/engine.py` severity tuning |

---

## Production upgrade checklist

- [ ] Replace `InMemoryBus` with Redpanda consumer groups
- [ ] Replace `DigitalTwinStore` with Redis + RedisTimeSeries
- [ ] Replace `AlertStore` + `WorkOrderStore` with TimescaleDB + PostGIS
- [ ] Enable real MQTT/TLS ingestion in `edge/gateway.py`
- [ ] Wire Twilio / FCM / SendGrid in `notify/service.py`
- [ ] Add Prometheus metrics + Grafana dashboards
- [ ] Deploy on GKE Autopilot via Helm (see `configs/helm/`)
- [ ] Enable GitHub Actions ML quality gate (F1 ≥ 0.95 gate)
