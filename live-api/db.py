"""
Database layer for the Omni-Sense live API.
Uses asyncpg to talk to TimescaleDB.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "timescaledb")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_NAME = os.getenv("DB_NAME", "omnisense")

# Global connection pool (_pool is created once at startup)
_pool: Optional[asyncpg.Pool] = None


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------
async def get_pool() -> asyncpg.Pool:
    """Return the existing pool or raise RuntimeError."""
    if _pool is None:
        raise RuntimeError("DB pool is not initialised")
    return _pool


async def init_db() -> None:
    """Create the connection pool and initialise tables."""
    global _pool
    _pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        min_size=2,
        max_size=10,
        command_timeout=60,
    )
    await _create_tables()


async def close_db() -> None:
    """Gracefully close the pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def db_health() -> bool:
    """Quick liveness check."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
async def _create_tables() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        # inference_results – hypertable
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_results (
                id          BIGSERIAL PRIMARY KEY,
                sensor_id   TEXT NOT NULL,
                timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                verdict     TEXT NOT NULL,
                confidence  DOUBLE PRECISION NOT NULL,
                probs       JSONB,
                features    JSONB,
                latency_ms  DOUBLE PRECISION,
                source      TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        # Make it a hypertable if not already
        try:
            await conn.execute(
                """
                SELECT create_hypertable('inference_results', 'timestamp',
                                         if_not_exists => TRUE,
                                         migrate_data => TRUE);
                """
            )
        except asyncpg.exceptions.UniqueViolationError:
            pass  # already a hypertable

        # tickets
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id          BIGSERIAL PRIMARY KEY,
                sensor_id   TEXT NOT NULL,
                verdict     TEXT NOT NULL,
                confidence  DOUBLE PRECISION NOT NULL,
                severity    TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'open',
                notes       TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                resolved_at TIMESTAMPTZ,
                resolution  TEXT,
                false_alarm BOOLEAN
            );
            """
        )

        # feedback
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id            BIGSERIAL PRIMARY KEY,
                inference_id  BIGINT NOT NULL,
                false_alarm   BOOLEAN NOT NULL,
                correct_verdict TEXT,
                notes         TEXT,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )

        # Indices for common queries
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_inference_sensor
                ON inference_results (sensor_id, timestamp DESC);
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_status
                ON tickets (status, created_at DESC);
            """
        )


# ---------------------------------------------------------------------------
# Inference results
# ---------------------------------------------------------------------------
async def insert_inference(
    sensor_id: str,
    verdict: str,
    confidence: float,
    probs: Optional[dict] = None,
    features: Optional[dict] = None,
    latency_ms: Optional[float] = None,
    source: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> int:
    """Insert a new inference result and return its id."""
    pool = await get_pool()
    ts = timestamp or datetime.now(timezone.utc)
    row_id: int = await pool.fetchval(
        """
        INSERT INTO inference_results
            (sensor_id, timestamp, verdict, confidence, probs, features, latency_ms, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id;
        """,
        sensor_id,
        ts,
        verdict,
        confidence,
        json.dumps(probs) if probs else None,
        json.dumps(features) if features else None,
        latency_ms,
        source,
    )
    return row_id


async def get_latest_inference(sensor_id: Optional[str] = None) -> Optional[dict]:
    """Return the single newest inference row (optionally filtered by sensor)."""
    pool = await get_pool()
    if sensor_id:
        row = await pool.fetchrow(
            """
            SELECT id, sensor_id, timestamp, verdict, confidence, probs, features,
                   latency_ms, source, created_at
            FROM inference_results
            WHERE sensor_id = $1
            ORDER BY timestamp DESC
            LIMIT 1;
            """,
            sensor_id,
        )
    else:
        row = await pool.fetchrow(
            """
            SELECT id, sensor_id, timestamp, verdict, confidence, probs, features,
                   latency_ms, source, created_at
            FROM inference_results
            ORDER BY timestamp DESC
            LIMIT 1;
            """
        )
    return dict(row) if row else None


async def get_inference_history(
    limit: int = 50, sensor_id: Optional[str] = None
) -> list[dict]:
    """Return recent inference rows."""
    pool = await get_pool()
    limit = max(1, min(limit, 10_000))
    if sensor_id:
        rows = await pool.fetch(
            """
            SELECT id, sensor_id, timestamp, verdict, confidence, probs, features,
                   latency_ms, source, created_at
            FROM inference_results
            WHERE sensor_id = $1
            ORDER BY timestamp DESC
            LIMIT $2;
            """,
            sensor_id,
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, sensor_id, timestamp, verdict, confidence, probs, features,
                   latency_ms, source, created_at
            FROM inference_results
            ORDER BY timestamp DESC
            LIMIT $1;
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def get_all_inferences() -> list[dict]:
    """Return every row – used for CSV export."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, sensor_id, timestamp, verdict, confidence, probs, features,
               latency_ms, source, created_at
        FROM inference_results
        ORDER BY timestamp DESC;
        """
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------
async def insert_ticket(
    sensor_id: str,
    verdict: str,
    confidence: float,
    severity: str,
    notes: Optional[str] = None,
) -> int:
    pool = await get_pool()
    row_id: int = await pool.fetchval(
        """
        INSERT INTO tickets (sensor_id, verdict, confidence, severity, status, notes)
        VALUES ($1, $2, $3, $4, 'open', $5)
        RETURNING id;
        """,
        sensor_id,
        verdict,
        confidence,
        severity,
        notes,
    )
    return row_id


async def get_tickets(status: Optional[str] = None) -> list[dict]:
    pool = await get_pool()
    if status:
        rows = await pool.fetch(
            """
            SELECT id, sensor_id, verdict, confidence, severity, status, notes,
                   created_at, resolved_at, resolution, false_alarm
            FROM tickets
            WHERE status = $1
            ORDER BY created_at DESC;
            """,
            status,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, sensor_id, verdict, confidence, severity, status, notes,
                   created_at, resolved_at, resolution, false_alarm
            FROM tickets
            ORDER BY created_at DESC;
            """
        )
    return [dict(r) for r in rows]


async def get_ticket_by_id(ticket_id: int) -> Optional[dict]:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT id, sensor_id, verdict, confidence, severity, status, notes,
               created_at, resolved_at, resolution, false_alarm
        FROM tickets
        WHERE id = $1;
        """,
        ticket_id,
    )
    return dict(row) if row else None


async def resolve_ticket(
    ticket_id: int,
    resolution: str,
    false_alarm: bool,
    notes: Optional[str] = None,
) -> bool:
    pool = await get_pool()
    result = await pool.execute(
        """
        UPDATE tickets
        SET status = 'closed',
            resolved_at = NOW(),
            resolution = $2,
            false_alarm = $3,
            notes = COALESCE($4, notes)
        WHERE id = $1 AND status = 'open';
        """,
        ticket_id,
        resolution,
        false_alarm,
        notes,
    )
    # asyncpg execute returns e.g. 'UPDATE 1'
    return result.strip().endswith("1")


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
async def insert_feedback(
    inference_id: int,
    false_alarm: bool,
    correct_verdict: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    pool = await get_pool()
    row_id: int = await pool.fetchval(
        """
        INSERT INTO feedback (inference_id, false_alarm, correct_verdict, notes)
        VALUES ($1, $2, $3, $4)
        RETURNING id;
        """,
        inference_id,
        false_alarm,
        correct_verdict,
        notes,
    )
    return row_id


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
async def get_system_metrics() -> dict[str, Any]:
    pool = await get_pool()
    inference_count = await pool.fetchval(
        "SELECT COUNT(*) FROM inference_results;"
    )
    alert_count = await pool.fetchval(
        "SELECT COUNT(*) FROM tickets WHERE status = 'open';"
    )
    avg_latency = await pool.fetchval(
        "SELECT AVG(latency_ms) FROM inference_results WHERE latency_ms IS NOT NULL;"
    )
    return {
        "inference_count": inference_count or 0,
        "alert_count": alert_count or 0,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0.0,
    }
