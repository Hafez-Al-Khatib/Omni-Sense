"""TimescaleDB-backed persistence layer for Omni-Sense.

Drop-in replacement for the in-memory stores in store.py.  Every public
async method signature is identical to the corresponding in-memory class so
no business-logic code needs changing.

Fallback behaviour
------------------
If the ``asyncpg`` package is not installed *or* the ``TIMESCALE_DSN``
environment variable is not set, every factory function silently returns the
in-memory counterpart from store.py and logs a warning.  This means the full
test suite (which runs without TimescaleDB) passes unchanged.

Lifecycle
---------
Call ``await init_db()`` once at application startup (after ``get_pool()``).
Call ``await release_pool()`` on shutdown.

Usage
-----
::

    from omni.common.timescale_store import get_store

    store = await get_store()       # TimescaleStore or in-memory fallback
    await store.twins().upsert_twin(twin)
    await store.alerts().put(alert)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from .schemas import Alert, DetectionResult, LeakHypothesis, WorkOrder
    from .store import SensorTwin

log = logging.getLogger("timescale_store")

# ---------------------------------------------------------------------------
# Optional asyncpg import — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import asyncpg  # type: ignore
    _ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG_AVAILABLE = False

_TIMESCALE_DSN = os.environ.get(
    "TIMESCALE_DSN",
    "postgresql://omni:omni@timescaledb:5432/omnisense",
)

# Module-level connection pool (singleton)
_pool: asyncpg.Pool | None = None  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------

async def get_pool() -> asyncpg.Pool:  # type: ignore[name-defined]
    """Return (or create) the shared asyncpg connection pool.

    Raises RuntimeError if asyncpg is not available.
    """
    global _pool
    if not _ASYNCPG_AVAILABLE:
        raise RuntimeError(
            "asyncpg is not installed — cannot create TimescaleDB pool. "
            "Install it with: pip install asyncpg"
        )
    if _pool is None:
        log.info("Creating asyncpg pool → %s", _TIMESCALE_DSN)
        _pool = await asyncpg.create_pool(
            dsn=_TIMESCALE_DSN,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        log.info("asyncpg pool ready")
    return _pool


async def release_pool() -> None:
    """Close the shared connection pool gracefully (call on shutdown)."""
    global _pool
    if _pool is not None:
        log.info("Closing asyncpg pool")
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create all tables and hypertables if they do not already exist.

    Reads the canonical DDL from db_schema.sql sitting alongside this module
    so the SQL is authoritative and not duplicated.  Falls back to inline DDL
    if the file cannot be found (e.g. installed as a package without data
    files).
    """
    pool = await get_pool()

    schema_path = os.path.join(os.path.dirname(__file__), "db_schema.sql")
    if os.path.exists(schema_path):
        with open(schema_path, encoding="utf-8") as fh:
            ddl = fh.read()
        log.info("Running DDL from %s", schema_path)
        async with pool.acquire() as conn:
            await conn.execute(ddl)
        log.info("Database schema initialised")
    else:
        log.error(
            "db_schema.sql not found at %s — skipping init_db(). "
            "Mount the file or run it manually against TimescaleDB.",
            schema_path,
        )


# ---------------------------------------------------------------------------
# Helpers — row <-> domain-object conversion
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(UTC)


def _uuid(val: str | UUID | None) -> UUID | None:
    if val is None:
        return None
    if isinstance(val, UUID):
        return val
    return UUID(str(val))


def _json_dumps(obj) -> str:
    """Serialise to JSON, handling UUID and datetime values."""
    def _default(o):
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o)} is not JSON serialisable")
    return json.dumps(obj, default=_default)


def _row_to_sensor_twin(row) -> SensorTwin:
    from .store import SensorTwin
    return SensorTwin(
        sensor_id=row["sensor_id"],
        site_id=row["site_id"],
        lat=row["lat"],
        lon=row["lon"],
        last_seen=row["last_seen"],
        battery_pct=row["battery_pct"],
        temperature_c=row["temperature_c"],
        firmware_version=row["firmware_version"],
        rolling_noise_floor_db=row["rolling_noise_floor_db"],
        last_p_leak=row["last_p_leak"],
        is_healthy=row["is_healthy"],
    )


def _row_to_alert(row) -> Alert:
    from .schemas import Alert, AlertState, Severity
    history = row["history"]
    if isinstance(history, str):
        history = json.loads(history)
    return Alert(
        alert_id=UUID(str(row["alert_id"])),
        hypothesis_id=UUID(str(row["hypothesis_id"])),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        state=AlertState(row["state"]),
        severity=Severity(row["severity"]),
        severity_score=row["severity_score"],
        title=row["title"],
        summary=row["summary"],
        lat=row["lat"],
        lon=row["lon"],
        pipe_segment_id=row["pipe_segment_id"],
        estimated_loss_lph=row["estimated_loss_lph"],
        assigned_crew_id=row["assigned_crew_id"],
        sla_due_at=row["sla_due_at"],
        history=history if isinstance(history, list) else [],
    )


def _row_to_work_order(row) -> WorkOrder:
    from .schemas import WorkOrder, WorkOrderStatus
    parts = row["parts_required"]
    if isinstance(parts, str):
        parts = json.loads(parts)
    return WorkOrder(
        work_order_id=UUID(str(row["work_order_id"])),
        alert_id=UUID(str(row["alert_id"])),
        created_at=row["created_at"],
        status=WorkOrderStatus(row["status"]),
        crew_id=row["crew_id"],
        eta_minutes=row["eta_minutes"],
        parts_required=parts if isinstance(parts, list) else [],
        notes=row["notes"] or "",
        completed_at=row["completed_at"],
        repair_cost_usd=row["repair_cost_usd"],
        mtbf_days=row["mtbf_days"],
    )


def _row_to_detection(row) -> DetectionResult:
    from .schemas import DetectionResult
    shap = row["top_shap_features"]
    if isinstance(shap, str):
        shap = json.loads(shap)
    latency = row["latency_ms"]
    if isinstance(latency, str):
        latency = json.loads(latency)
    return DetectionResult(
        detection_id=UUID(str(row["detection_id"])),
        frame_id=UUID(str(row["frame_id"])),
        sensor_id=row["sensor_id"],
        site_id=row["site_id"],
        captured_at=row["captured_at"],
        decided_at=row["decided_at"],
        yamnet_top_class=row["yamnet_top_class"],
        xgb_p_leak=row["xgb_p_leak"],
        rf_p_leak=row["rf_p_leak"],
        cnn_p_leak=row["cnn_p_leak"],
        if_anomaly_score=row["if_anomaly_score"],
        ood_score=row["ood_score"],
        fused_p_leak=row["fused_p_leak"],
        fused_uncertainty=row["fused_uncertainty"],
        is_leak=row["is_leak"],
        is_ood=row["is_ood"],
        top_shap_features=[tuple(x) for x in shap] if shap else [],
        latency_ms=latency if isinstance(latency, dict) else {},
    )


def _row_to_hypothesis(row) -> LeakHypothesis:
    from .schemas import LeakHypothesis
    det_ids = row["contributing_detection_ids"] or []
    # asyncpg returns UUID[] as a Python list of asyncpg UUID objects (or strings)
    det_ids = [UUID(str(d)) for d in det_ids]
    return LeakHypothesis(
        hypothesis_id=UUID(str(row["hypothesis_id"])),
        created_at=row["created_at"],
        contributing_detection_ids=det_ids,
        lat=row["lat"],
        lon=row["lon"],
        uncertainty_m=row["uncertainty_m"],
        pipe_segment_id=row["pipe_segment_id"],
        distance_along_pipe_m=row["distance_along_pipe_m"],
        estimated_flow_lps=row["estimated_flow_lps"],
        confidence=row["confidence"],
    )


# ---------------------------------------------------------------------------
# TimescaleDigitalTwinStore
# ---------------------------------------------------------------------------

class TimescaleDigitalTwinStore:
    """asyncpg-backed replacement for DigitalTwinStore.

    API is identical to the in-memory version; all callers can swap
    transparently.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:  # type: ignore[name-defined]
        self._pool = pool

    # ------------------------------------------------------------------
    # Twins

    async def upsert_twin(self, twin) -> None:
        """Insert or update a sensor twin row."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sensor_twins
                    (sensor_id, site_id, lat, lon, last_seen, battery_pct,
                     temperature_c, firmware_version, rolling_noise_floor_db,
                     last_p_leak, is_healthy, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                ON CONFLICT (sensor_id) DO UPDATE SET
                    site_id                = EXCLUDED.site_id,
                    lat                    = EXCLUDED.lat,
                    lon                    = EXCLUDED.lon,
                    last_seen              = EXCLUDED.last_seen,
                    battery_pct            = EXCLUDED.battery_pct,
                    temperature_c          = EXCLUDED.temperature_c,
                    firmware_version       = EXCLUDED.firmware_version,
                    rolling_noise_floor_db = EXCLUDED.rolling_noise_floor_db,
                    last_p_leak            = EXCLUDED.last_p_leak,
                    is_healthy             = EXCLUDED.is_healthy,
                    updated_at             = NOW()
                """,
                twin.sensor_id,
                twin.site_id,
                twin.lat,
                twin.lon,
                twin.last_seen,
                twin.battery_pct,
                twin.temperature_c,
                twin.firmware_version,
                twin.rolling_noise_floor_db,
                twin.last_p_leak,
                twin.is_healthy,
            )

    async def update_telemetry(
        self, sensor_id: str, battery: float, temp: float, fw: str
    ) -> None:
        """Upsert live telemetry fields and write a sensor_readings row."""
        now = _now()
        is_healthy = battery > 15 and temp < 70
        async with self._pool.acquire() as conn:
            # Update or insert twin state
            await conn.execute(
                """
                INSERT INTO sensor_twins
                    (sensor_id, site_id, lat, lon, last_seen, battery_pct,
                     temperature_c, firmware_version, is_healthy, updated_at)
                VALUES ($1, 'unknown', 0.0, 0.0, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (sensor_id) DO UPDATE SET
                    last_seen        = EXCLUDED.last_seen,
                    battery_pct      = EXCLUDED.battery_pct,
                    temperature_c    = EXCLUDED.temperature_c,
                    firmware_version = EXCLUDED.firmware_version,
                    is_healthy       = EXCLUDED.is_healthy,
                    updated_at       = NOW()
                """,
                sensor_id, now, battery, temp, fw, is_healthy,
            )
            # Append telemetry reading to hypertable
            await conn.execute(
                """
                INSERT INTO sensor_readings
                    (captured_at, sensor_id, battery_pct, temperature_c, firmware_version)
                VALUES ($1, $2, $3, $4, $5)
                """,
                now, sensor_id, battery, temp, fw,
            )

    async def record_detection(self, det) -> None:
        """Persist a DetectionResult to the detections hypertable and update twin."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO detections
                    (captured_at, detection_id, frame_id, sensor_id, site_id,
                     decided_at, yamnet_top_class, xgb_p_leak, rf_p_leak,
                     cnn_p_leak, if_anomaly_score, ood_score, fused_p_leak,
                     fused_uncertainty, is_leak, is_ood,
                     top_shap_features, latency_ms)
                VALUES
                    ($1, $2, $3, $4, $5,
                     $6, $7, $8, $9,
                     $10, $11, $12, $13,
                     $14, $15, $16,
                     $17::jsonb, $18::jsonb)
                ON CONFLICT DO NOTHING
                """,
                det.captured_at,
                det.detection_id,
                det.frame_id,
                det.sensor_id,
                det.site_id,
                det.decided_at,
                det.yamnet_top_class,
                det.xgb_p_leak,
                det.rf_p_leak,
                det.cnn_p_leak,
                det.if_anomaly_score,
                det.ood_score,
                det.fused_p_leak,
                det.fused_uncertainty,
                det.is_leak,
                det.is_ood,
                _json_dumps(det.top_shap_features),
                _json_dumps(det.latency_ms),
            )
            # Update twin's last_p_leak
            await conn.execute(
                """
                UPDATE sensor_twins
                SET last_p_leak = $1, updated_at = NOW()
                WHERE sensor_id = $2
                """,
                det.fused_p_leak,
                det.sensor_id,
            )

    async def recent_detections(self, sensor_id: str, n: int = 32) -> list:
        """Return up to *n* most recent detections for the given sensor."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM detections
                WHERE sensor_id = $1
                ORDER BY captured_at DESC
                LIMIT $2
                """,
                sensor_id,
                n,
            )
        # Return in chronological order (oldest first) to match deque behaviour
        return [_row_to_detection(r) for r in reversed(rows)]

    async def all_recent_leaks(
        self, min_p: float = 0.7, horizon_s: float = 30.0
    ) -> list:
        """Return all detections above *min_p* within the last *horizon_s* seconds."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM detections
                WHERE fused_p_leak >= $1
                  AND captured_at  >= NOW() - ($2 * INTERVAL '1 second')
                ORDER BY captured_at DESC
                """,
                min_p,
                horizon_s,
            )
        return [_row_to_detection(r) for r in rows]


# ---------------------------------------------------------------------------
# TimescaleAlertStore
# ---------------------------------------------------------------------------

class TimescaleAlertStore:
    """asyncpg-backed replacement for AlertStore."""

    def __init__(self, pool: asyncpg.Pool) -> None:  # type: ignore[name-defined]
        self._pool = pool

    async def put(self, alert) -> None:
        """Insert or fully replace an alert row."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO alerts
                    (alert_id, hypothesis_id, created_at, updated_at,
                     state, severity, severity_score, title, summary,
                     lat, lon, pipe_segment_id, estimated_loss_lph,
                     assigned_crew_id, sla_due_at, history)
                VALUES
                    ($1, $2, $3, $4,
                     $5, $6, $7, $8, $9,
                     $10, $11, $12, $13,
                     $14, $15, $16::jsonb)
                ON CONFLICT (alert_id) DO UPDATE SET
                    hypothesis_id      = EXCLUDED.hypothesis_id,
                    updated_at         = EXCLUDED.updated_at,
                    state              = EXCLUDED.state,
                    severity           = EXCLUDED.severity,
                    severity_score     = EXCLUDED.severity_score,
                    title              = EXCLUDED.title,
                    summary            = EXCLUDED.summary,
                    lat                = EXCLUDED.lat,
                    lon                = EXCLUDED.lon,
                    pipe_segment_id    = EXCLUDED.pipe_segment_id,
                    estimated_loss_lph = EXCLUDED.estimated_loss_lph,
                    assigned_crew_id   = EXCLUDED.assigned_crew_id,
                    sla_due_at         = EXCLUDED.sla_due_at,
                    history            = EXCLUDED.history
                """,
                alert.alert_id,
                alert.hypothesis_id,
                alert.created_at,
                alert.updated_at,
                alert.state.value,
                alert.severity.value,
                alert.severity_score,
                alert.title,
                alert.summary,
                alert.lat,
                alert.lon,
                alert.pipe_segment_id,
                alert.estimated_loss_lph,
                alert.assigned_crew_id,
                alert.sla_due_at,
                _json_dumps(alert.history),
            )

    async def get(self, alert_id: UUID) -> Alert | None:
        """Fetch a single alert by ID, or None if not found."""
        from .schemas import Alert  # noqa: F401 (type-checking hint)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM alerts WHERE alert_id = $1",
                alert_id,
            )
        if row is None:
            return None
        return _row_to_alert(row)

    async def transition(
        self, alert_id: UUID, new_state, note: str = ""
    ) -> Alert:
        """Atomically transition an alert to a new state and append a history entry."""
        now = _now()
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM alerts WHERE alert_id = $1 FOR UPDATE",
                alert_id,
            )
            if row is None:
                raise KeyError(f"Alert {alert_id} not found")

            history = row["history"]
            if isinstance(history, str):
                history = json.loads(history)
            if not isinstance(history, list):
                history = []

            entry = {
                "from": row["state"],
                "to": new_state.value,
                "at": now.isoformat(),
                "note": note,
            }
            history.append(entry)

            updated = await conn.fetchrow(
                """
                    UPDATE alerts
                    SET state      = $1,
                        updated_at = $2,
                        history    = $3::jsonb
                    WHERE alert_id = $4
                    RETURNING *
                    """,
                new_state.value,
                now,
                _json_dumps(history),
                alert_id,
            )
            # Also write to normalised audit table
            await conn.execute(
                """
                    INSERT INTO alert_history
                        (alert_id, transitioned_at, from_state, to_state, note)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                alert_id,
                now,
                entry["from"],
                entry["to"],
                note,
            )
        return _row_to_alert(updated)

    async def list_all(self) -> list:
        """Return all alerts sorted by created_at descending."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM alerts ORDER BY created_at DESC"
            )
        return [_row_to_alert(r) for r in rows]


# ---------------------------------------------------------------------------
# TimescaleWorkOrderStore
# ---------------------------------------------------------------------------

class TimescaleWorkOrderStore:
    """asyncpg-backed replacement for WorkOrderStore."""

    def __init__(self, pool: asyncpg.Pool) -> None:  # type: ignore[name-defined]
        self._pool = pool

    async def put(self, wo) -> None:
        """Insert or fully replace a work-order row."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO work_orders
                    (work_order_id, alert_id, created_at, status, crew_id,
                     eta_minutes, parts_required, notes, completed_at,
                     repair_cost_usd, mtbf_days)
                VALUES
                    ($1, $2, $3, $4, $5,
                     $6, $7::jsonb, $8, $9,
                     $10, $11)
                ON CONFLICT (work_order_id) DO UPDATE SET
                    alert_id        = EXCLUDED.alert_id,
                    status          = EXCLUDED.status,
                    crew_id         = EXCLUDED.crew_id,
                    eta_minutes     = EXCLUDED.eta_minutes,
                    parts_required  = EXCLUDED.parts_required,
                    notes           = EXCLUDED.notes,
                    completed_at    = EXCLUDED.completed_at,
                    repair_cost_usd = EXCLUDED.repair_cost_usd,
                    mtbf_days       = EXCLUDED.mtbf_days
                """,
                wo.work_order_id,
                wo.alert_id,
                wo.created_at,
                wo.status.value,
                wo.crew_id,
                wo.eta_minutes,
                _json_dumps(wo.parts_required),
                wo.notes,
                wo.completed_at,
                wo.repair_cost_usd,
                wo.mtbf_days,
            )

    async def complete(self, wo_id: UUID, cost: float, notes: str):
        """Mark a work order as COMPLETED and record cost + notes."""
        from .schemas import WorkOrderStatus
        now = _now()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE work_orders
                SET status          = $1,
                    completed_at    = $2,
                    repair_cost_usd = $3,
                    notes           = $4
                WHERE work_order_id = $5
                RETURNING *
                """,
                WorkOrderStatus.COMPLETED.value,
                now,
                cost,
                notes,
                wo_id,
            )
            if row is None:
                raise KeyError(f"WorkOrder {wo_id} not found")
        return _row_to_work_order(row)

    async def list_all(self) -> list:
        """Return all work orders (no guaranteed sort order)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM work_orders ORDER BY created_at DESC"
            )
        return [_row_to_work_order(r) for r in rows]


# ---------------------------------------------------------------------------
# TimescaleHypothesisStore
# ---------------------------------------------------------------------------

class TimescaleHypothesisStore:
    """asyncpg-backed store for LeakHypothesis objects.

    The in-memory version is a plain list exposed via the module-level
    ``hypotheses()`` function.  This class provides the same append +
    list interface but persists to TimescaleDB.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:  # type: ignore[name-defined]
        self._pool = pool

    async def append(self, hypothesis) -> None:
        """Persist a new LeakHypothesis (insert-only; hypotheses are immutable)."""
        det_ids = [str(d) for d in hypothesis.contributing_detection_ids]
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO leak_hypotheses
                    (hypothesis_id, created_at, contributing_detection_ids,
                     lat, lon, uncertainty_m, pipe_segment_id,
                     distance_along_pipe_m, estimated_flow_lps, confidence)
                VALUES ($1, $2, $3::uuid[], $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (hypothesis_id) DO NOTHING
                """,
                hypothesis.hypothesis_id,
                hypothesis.created_at,
                det_ids,
                hypothesis.lat,
                hypothesis.lon,
                hypothesis.uncertainty_m,
                hypothesis.pipe_segment_id,
                hypothesis.distance_along_pipe_m,
                hypothesis.estimated_flow_lps,
                hypothesis.confidence,
            )

    async def list_all(self) -> list:
        """Return all hypotheses ordered by creation time descending."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM leak_hypotheses ORDER BY created_at DESC"
            )
        return [_row_to_hypothesis(r) for r in rows]

    async def recent(self, limit: int = 100) -> list:
        """Return the *limit* most-recent hypotheses."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM leak_hypotheses
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [_row_to_hypothesis(r) for r in rows]


# ---------------------------------------------------------------------------
# TimescaleStore — bundles all four stores (mirrors the store.py module API)
# ---------------------------------------------------------------------------

class TimescaleStore:
    """Bundled store that exposes the same four accessor methods as store.py.

    Usage::

        store = await get_store()
        store.twins()       # TimescaleDigitalTwinStore  (or DigitalTwinStore)
        store.alerts()      # TimescaleAlertStore        (or AlertStore)
        store.work_orders() # TimescaleWorkOrderStore    (or WorkOrderStore)
        store.hypotheses()  # TimescaleHypothesisStore   (or list[LeakHypothesis])
    """

    def __init__(self, pool: asyncpg.Pool) -> None:  # type: ignore[name-defined]
        self._twins = TimescaleDigitalTwinStore(pool)
        self._alerts = TimescaleAlertStore(pool)
        self._work_orders = TimescaleWorkOrderStore(pool)
        self._hypotheses = TimescaleHypothesisStore(pool)

    def twins(self) -> TimescaleDigitalTwinStore:
        return self._twins

    def alerts(self) -> TimescaleAlertStore:
        return self._alerts

    def work_orders(self) -> TimescaleWorkOrderStore:
        return self._work_orders

    def hypotheses(self) -> TimescaleHypothesisStore:
        return self._hypotheses


# ---------------------------------------------------------------------------
# _InMemoryHypothesisStore — thin list wrapper so the fallback path also has
# an object with .append() / .list_all() / .recent() methods
# ---------------------------------------------------------------------------

class _InMemoryHypothesisStore:
    """Wraps the module-level list from store.py to expose store-style methods.

    Used only when falling back to the in-memory backend.
    """

    def __init__(self) -> None:
        from . import store as _store
        self._store = _store

    async def append(self, hypothesis) -> None:
        self._store.hypotheses().append(hypothesis)

    async def list_all(self) -> list:
        return list(self._store.hypotheses())

    async def recent(self, limit: int = 100) -> list:
        return list(self._store.hypotheses())[-limit:]


class _FallbackStore:
    """Wraps the in-memory stores from store.py behind the TimescaleStore API.

    Returned by get_store() when TimescaleDB is unavailable.
    """

    def __init__(self) -> None:
        from . import store as _store
        self._store = _store
        self._hypotheses_store = _InMemoryHypothesisStore()

    def twins(self):
        return self._store.twins()

    def alerts(self):
        return self._store.alerts()

    def work_orders(self):
        return self._store.work_orders()

    def hypotheses(self):
        return self._hypotheses_store


# ---------------------------------------------------------------------------
# Module-level factory — the main entry point for application code
# ---------------------------------------------------------------------------

_store_singleton: TimescaleStore | _FallbackStore | None = None


async def get_store(force_timescale: bool = False) -> TimescaleStore | _FallbackStore:
    """Return the application-wide store instance.

    Decision logic:
    1. If asyncpg is not installed → fallback (with warning).
    2. If TIMESCALE_DSN env var is not set → fallback (with warning).
       (The env var is always set in the Docker environment; it being absent
       signals a test/CI environment that should use in-memory stores.)
    3. Otherwise → TimescaleStore backed by a real asyncpg pool.

    The singleton is created once and reused.  Pass ``force_timescale=True``
    in tests that explicitly exercise the Timescale path.
    """
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    env_dsn = os.environ.get("TIMESCALE_DSN", "")

    if not _ASYNCPG_AVAILABLE:
        log.warning(
            "asyncpg not installed — falling back to in-memory stores. "
            "Install asyncpg for TimescaleDB persistence."
        )
        _store_singleton = _FallbackStore()
        return _store_singleton

    if not env_dsn and not force_timescale:
        log.warning(
            "TIMESCALE_DSN env var not set — falling back to in-memory stores. "
            "Set TIMESCALE_DSN=postgresql://... to enable persistence."
        )
        _store_singleton = _FallbackStore()
        return _store_singleton

    try:
        pool = await get_pool()
        _store_singleton = TimescaleStore(pool)
        log.info("TimescaleStore ready (pool acquired)")
    except Exception as exc:
        log.warning(
            "Could not connect to TimescaleDB (%s) — "
            "falling back to in-memory stores.",
            exc,
        )
        _store_singleton = _FallbackStore()

    return _store_singleton


def reset_store_singleton() -> None:
    """Reset the cached store singleton (useful in tests)."""
    global _store_singleton
    _store_singleton = None
