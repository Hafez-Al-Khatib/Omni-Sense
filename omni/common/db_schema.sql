-- =============================================================================
-- Omni-Sense Platform — Full Database Schema
-- TimescaleDB + PostGIS
--
-- Run order: this file is mounted as /docker-entrypoint-initdb.d/01_schema.sql
-- so PostgreSQL will execute it automatically on first container start.
--
-- Tables:
--   sensor_twins       — live digital-twin state for each edge sensor (regular)
--   sensor_readings    — telemetry hypertable (time-series, partitioned by captured_at)
--   detections         — inference results hypertable (time-series)
--   alerts             — leak alert lifecycle records (regular)
--   alert_history      — state-transition journal for alerts (append-only)
--   work_orders        — field-crew work-order records (regular)
--   leak_hypotheses    — fused spatial hypotheses (regular, lat/lon indexed)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
-- PostGIS: required for future geo queries (pipe-network routing, spatial joins)
CREATE EXTENSION IF NOT EXISTS postgis CASCADE;

-- ---------------------------------------------------------------------------
-- sensor_twins  — one row per physical sensor; updated in-place (upsert)
--
-- Stores the "live" digital-twin state that the platform maintains between
-- acoustic frames.  Not a time-series table because we only need the most
-- recent state; historical telemetry goes into sensor_readings.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sensor_twins (
    sensor_id               TEXT        PRIMARY KEY,
    site_id                 TEXT        NOT NULL DEFAULT 'unknown',
    lat                     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    lon                     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    last_seen               TIMESTAMPTZ,
    battery_pct             DOUBLE PRECISION NOT NULL DEFAULT 100.0,
    temperature_c           DOUBLE PRECISION NOT NULL DEFAULT 25.0,
    firmware_version        TEXT        NOT NULL DEFAULT 'unknown',
    rolling_noise_floor_db  DOUBLE PRECISION NOT NULL DEFAULT -60.0,
    last_p_leak             DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    is_healthy              BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE sensor_twins IS
    'Live digital-twin state for every registered edge sensor. '
    'Upserted on each telemetry heartbeat or detection result.';

-- Spatial index on (lat, lon) for proximity queries
CREATE INDEX IF NOT EXISTS idx_sensor_twins_site
    ON sensor_twins (site_id);

-- ---------------------------------------------------------------------------
-- sensor_readings  — hypertable for edge telemetry (battery, temp, disk …)
--
-- Partitioned by captured_at with a 1-day chunk interval so each day of
-- readings lives in its own chunk — enables fast time-range scans and
-- efficient data retention policies.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sensor_readings (
    captured_at             TIMESTAMPTZ NOT NULL,
    sensor_id               TEXT        NOT NULL,
    battery_pct             DOUBLE PRECISION,
    temperature_c           DOUBLE PRECISION,
    disk_free_mb            DOUBLE PRECISION,
    rtc_drift_ms            INTEGER,
    uptime_s                INTEGER,
    firmware_version        TEXT
);

-- Convert to hypertable; IF NOT EXISTS guard prevents errors on re-run
SELECT create_hypertable(
    'sensor_readings',
    'captured_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

COMMENT ON TABLE sensor_readings IS
    'TimescaleDB hypertable storing edge telemetry samples. '
    'One row per TelemetrySample published to the bus. '
    'Chunk interval: 1 day. Suitable for retention policies (e.g. 90 days).';

CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_time
    ON sensor_readings (sensor_id, captured_at DESC);

-- ---------------------------------------------------------------------------
-- detections  — hypertable for ML inference results
--
-- One row per DetectionResult.  The fused_p_leak and is_leak columns are the
-- primary targets for time-range aggregation dashboards.
-- Chunk interval: 1 hour — detections are high-frequency (up to 1 Hz per sensor).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS detections (
    captured_at             TIMESTAMPTZ NOT NULL,
    detection_id            UUID        NOT NULL,
    frame_id                UUID        NOT NULL,
    sensor_id               TEXT        NOT NULL,
    site_id                 TEXT        NOT NULL,
    decided_at              TIMESTAMPTZ NOT NULL,

    -- Per-model outputs
    yamnet_top_class        TEXT,
    xgb_p_leak              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    rf_p_leak               DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    cnn_p_leak              DOUBLE PRECISION,
    if_anomaly_score        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ood_score               DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    -- Fused outputs
    fused_p_leak            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    fused_uncertainty       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    is_leak                 BOOLEAN     NOT NULL DEFAULT FALSE,
    is_ood                  BOOLEAN     NOT NULL DEFAULT FALSE,

    -- Explainability (stored as JSONB array of [feature, shap_value] pairs)
    top_shap_features       JSONB       NOT NULL DEFAULT '[]',

    -- Latency breakdown (stored as JSONB object {stage: ms})
    latency_ms              JSONB       NOT NULL DEFAULT '{}'
);

SELECT create_hypertable(
    'detections',
    'captured_at',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists       => TRUE
);

COMMENT ON TABLE detections IS
    'TimescaleDB hypertable storing every ML inference result. '
    'Primary query pattern: WHERE sensor_id = $1 AND captured_at > NOW() - INTERVAL ''30s''. '
    'Chunk interval: 1 hour.';

CREATE INDEX IF NOT EXISTS idx_detections_sensor_time
    ON detections (sensor_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_detections_is_leak_time
    ON detections (is_leak, captured_at DESC)
    WHERE is_leak = TRUE;

CREATE INDEX IF NOT EXISTS idx_detections_fused_p_leak
    ON detections (fused_p_leak DESC, captured_at DESC);

-- ---------------------------------------------------------------------------
-- alerts  — lifecycle records for leak alerts raised by the hypothesis engine
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS alerts (
    alert_id                UUID        PRIMARY KEY,
    hypothesis_id           UUID        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    state                   TEXT        NOT NULL DEFAULT 'NEW',
    severity                TEXT        NOT NULL,
    severity_score          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    title                   TEXT        NOT NULL DEFAULT '',
    summary                 TEXT        NOT NULL DEFAULT '',
    lat                     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    lon                     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    pipe_segment_id         TEXT,
    estimated_loss_lph      DOUBLE PRECISION,
    assigned_crew_id        TEXT,
    sla_due_at              TIMESTAMPTZ,
    -- Full state-history stored as JSONB array; alert_history table provides
    -- a normalised view for audit queries.
    history                 JSONB       NOT NULL DEFAULT '[]'
);

COMMENT ON TABLE alerts IS
    'One row per raised alert.  The history JSONB column is the fast path '
    'for displaying the state timeline; alert_history is the audit table.';

CREATE INDEX IF NOT EXISTS idx_alerts_state
    ON alerts (state, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_created_at
    ON alerts (created_at DESC);

-- ---------------------------------------------------------------------------
-- alert_history  — append-only state-transition journal
--
-- Written by TimescaleAlertStore.transition() in addition to updating the
-- alerts.history JSONB column so that forensic/audit queries can use SQL
-- rather than JSONB path expressions.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS alert_history (
    id                      BIGSERIAL   PRIMARY KEY,
    alert_id                UUID        NOT NULL REFERENCES alerts(alert_id) ON DELETE CASCADE,
    transitioned_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    from_state              TEXT        NOT NULL,
    to_state                TEXT        NOT NULL,
    note                    TEXT        NOT NULL DEFAULT ''
);

COMMENT ON TABLE alert_history IS
    'Normalised, append-only audit log for alert state transitions.';

CREATE INDEX IF NOT EXISTS idx_alert_history_alert_id
    ON alert_history (alert_id, transitioned_at DESC);

-- ---------------------------------------------------------------------------
-- work_orders  — field-crew dispatch records
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS work_orders (
    work_order_id           UUID        PRIMARY KEY,
    alert_id                UUID        NOT NULL REFERENCES alerts(alert_id) ON DELETE RESTRICT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status                  TEXT        NOT NULL DEFAULT 'DRAFT',
    crew_id                 TEXT        NOT NULL,
    eta_minutes             INTEGER     NOT NULL DEFAULT 0,
    parts_required          JSONB       NOT NULL DEFAULT '[]',
    notes                   TEXT        NOT NULL DEFAULT '',
    completed_at            TIMESTAMPTZ,
    repair_cost_usd         DOUBLE PRECISION,
    mtbf_days               DOUBLE PRECISION
);

COMMENT ON TABLE work_orders IS
    'One row per field-crew work order. Created by the dispatch service when '
    'an alert is acknowledged and a crew is assigned.';

CREATE INDEX IF NOT EXISTS idx_work_orders_alert_id
    ON work_orders (alert_id);

CREATE INDEX IF NOT EXISTS idx_work_orders_status
    ON work_orders (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_work_orders_crew_id
    ON work_orders (crew_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- leak_hypotheses  — fused spatial hypotheses from multi-sensor correlation
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS leak_hypotheses (
    hypothesis_id               UUID        PRIMARY KEY,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- contributing_detection_ids stored as UUID array (native PG type)
    contributing_detection_ids  UUID[]      NOT NULL DEFAULT '{}',
    lat                         DOUBLE PRECISION NOT NULL,
    lon                         DOUBLE PRECISION NOT NULL,
    uncertainty_m               DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    pipe_segment_id             TEXT,
    distance_along_pipe_m       DOUBLE PRECISION,
    estimated_flow_lps          DOUBLE PRECISION,
    confidence                  DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

COMMENT ON TABLE leak_hypotheses IS
    'Fused spatial hypothesis produced by the Hypothesis Engine after '
    'correlating detections from multiple nearby sensors. '
    'Each hypothesis may trigger one Alert.';

CREATE INDEX IF NOT EXISTS idx_leak_hypotheses_created_at
    ON leak_hypotheses (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_leak_hypotheses_pipe_segment
    ON leak_hypotheses (pipe_segment_id)
    WHERE pipe_segment_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_leak_hypotheses_confidence
    ON leak_hypotheses (confidence DESC, created_at DESC);

-- ---------------------------------------------------------------------------
-- inference_results  — hypertable for vibration-analysis inference results
-- (populated by omni/edge/mqtt_bridge/bridge.py)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS inference_results (
    captured_at             TIMESTAMPTZ NOT NULL,
    sensor_id               TEXT        NOT NULL,
    verdict                 TEXT        NOT NULL,
    confidence              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    latency_ms              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    features                JSONB       NOT NULL DEFAULT '{}',
    source                  TEXT        NOT NULL DEFAULT 'vibration_analysis'
);

SELECT create_hypertable(
    'inference_results',
    'captured_at',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_inference_results_sensor_time
    ON inference_results (sensor_id, captured_at DESC);

-- ---------------------------------------------------------------------------
-- tickets  — support / maintenance tickets
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id               UUID        PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sensor_id               TEXT        NOT NULL,
    title                   TEXT        NOT NULL DEFAULT '',
    description             TEXT        NOT NULL DEFAULT '',
    status                  TEXT        NOT NULL DEFAULT 'OPEN',
    priority                TEXT        NOT NULL DEFAULT 'MEDIUM',
    assigned_to             TEXT,
    resolved_at             TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_tickets_sensor_id
    ON tickets (sensor_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tickets_status
    ON tickets (status, created_at DESC);

-- ---------------------------------------------------------------------------
-- feedback  — user feedback on inference results or alerts
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id             UUID        PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sensor_id               TEXT        NOT NULL,
    related_detection_id    UUID,
    rating                  INTEGER     CHECK (rating >= 1 AND rating <= 5),
    comment                 TEXT        NOT NULL DEFAULT '',
    category                TEXT        NOT NULL DEFAULT 'GENERAL'
);

CREATE INDEX IF NOT EXISTS idx_feedback_sensor_id
    ON feedback (sensor_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_feedback_detection_id
    ON feedback (related_detection_id)
    WHERE related_detection_id IS NOT NULL;
