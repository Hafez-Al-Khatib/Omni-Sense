"""Shared Pydantic schemas for Omni-Sense platform v2.

Every event on the bus validates against one of these. Versioned explicitly
so we can evolve without breaking consumers.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─────────────────────────── Edge ingestion ────────────────────────────
class AcousticFrame(BaseModel):
    """Raw-ish acoustic frame emitted by an edge sensor.

    The edge node applies VAD + gain normalization then publishes a 16 kHz,
    mono, PCM16 window of exactly 0.975 s (15600 samples), matching YAMNet's
    input window.
    """

    schema_version: Literal["1"] = "1"
    frame_id: UUID = Field(default_factory=uuid4)
    sensor_id: str
    site_id: str
    captured_at: datetime
    sample_rate_hz: int = 16000
    n_samples: int = 15600
    pcm_b64: str = Field(description="Base64-encoded little-endian PCM16")
    edge_snr_db: float
    edge_vad_confidence: float
    firmware_version: str


class TelemetrySample(BaseModel):
    """Edge health telemetry (battery, temp, disk, drift)."""

    schema_version: Literal["1"] = "1"
    sensor_id: str
    captured_at: datetime
    battery_pct: float
    temperature_c: float
    disk_free_mb: float
    rtc_drift_ms: int
    uptime_s: int
    firmware_version: str


# ─────────────────────────── Detection ────────────────────────────────
class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionResult(BaseModel):
    """Output of EEP v2 after fan-out to IEPs.

    Never write business rules in here — this is a pure inference record.
    """

    schema_version: Literal["1"] = "1"
    detection_id: UUID = Field(default_factory=uuid4)
    frame_id: UUID
    sensor_id: str
    site_id: str
    captured_at: datetime
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Per-head outputs
    yamnet_top_class: str | None = None
    xgb_p_leak: float = Field(ge=0.0, le=1.0)
    rf_p_leak: float = Field(ge=0.0, le=1.0)
    cnn_p_leak: float | None = None
    if_anomaly_score: float
    ood_score: float = Field(description="Deep-SVDD distance; >1.0 = out-of-distribution")

    # Fused
    fused_p_leak: float = Field(ge=0.0, le=1.0)
    fused_uncertainty: float = Field(ge=0.0, description="MC-Dropout std-dev")
    is_leak: bool
    is_ood: bool

    # Explainability
    top_shap_features: list[tuple[str, float]] = Field(default_factory=list)

    # Compute
    latency_ms: dict[str, float] = Field(default_factory=dict)


# ─────────────────────────── Spatial ──────────────────────────────────
class LeakHypothesis(BaseModel):
    """A fused spatial hypothesis from correlated detections across sensors."""

    schema_version: Literal["1"] = "1"
    hypothesis_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    contributing_detection_ids: list[UUID]
    lat: float
    lon: float
    uncertainty_m: float = Field(description="95% confidence radius in meters")
    pipe_segment_id: str | None = None
    distance_along_pipe_m: float | None = None
    estimated_flow_lps: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)


# ─────────────────────────── Alerts ───────────────────────────────────
class AlertState(StrEnum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISPATCHED = "DISPATCHED"
    ON_SITE = "ON_SITE"
    RESOLVED = "RESOLVED"
    VERIFIED = "VERIFIED"
    SUPPRESSED = "SUPPRESSED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class Alert(BaseModel):
    schema_version: Literal["1"] = "1"
    alert_id: UUID = Field(default_factory=uuid4)
    hypothesis_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state: AlertState = AlertState.NEW
    severity: Severity
    severity_score: float = Field(ge=0.0, le=100.0)
    title: str
    summary: str
    lat: float
    lon: float
    pipe_segment_id: str | None = None
    estimated_loss_lph: float | None = None
    assigned_crew_id: str | None = None
    sla_due_at: datetime | None = None
    history: list[dict] = Field(default_factory=list)


# ─────────────────────────── Work orders ──────────────────────────────
class WorkOrderStatus(StrEnum):
    DRAFT = "DRAFT"
    DISPATCHED = "DISPATCHED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class WorkOrder(BaseModel):
    schema_version: Literal["1"] = "1"
    work_order_id: UUID = Field(default_factory=uuid4)
    alert_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: WorkOrderStatus = WorkOrderStatus.DRAFT
    crew_id: str
    eta_minutes: int
    parts_required: list[str] = Field(default_factory=list)
    notes: str = ""
    completed_at: datetime | None = None
    repair_cost_usd: float | None = None
    mtbf_days: float | None = None


# ─────────────────────────── Audit ────────────────────────────────────
class AuditEvent(BaseModel):
    schema_version: Literal["1"] = "1"
    event_id: UUID = Field(default_factory=uuid4)
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str
    # Primary action field — use either action (legacy) or event_type (preferred)
    action: str | None = None
    event_type: str | None = None   # e.g. "mlops.retrain_triggered"
    resource_type: str | None = None
    resource_id: str | None = None
    details: dict | None = None     # arbitrary structured payload
    payload_hash_sha256: str = ""
    prev_hash: str = ""
    signature_ed25519: str = ""


# ──────────────────────────── SCADA ───────────────────────────────────────────
class ScadaReading(BaseModel):
    """A pressure/flow/temperature reading sourced from an OPC-UA SCADA server.

    Published on Topics.SCADA_READING ("scada.reading.v1") by the OPC-UA
    gateway (omni/edge/opcua_gateway.py).  IEP2's SCADA consistency check
    subscribes to this topic to correlate with acoustic detections.
    """

    schema_version: Literal["1"] = "1"
    sensor_id: str = Field(
        description="Omni-Sense sensor / asset ID that this reading maps to"
    )
    site_id: str = Field(
        default="",
        description="Site / zone identifier matching AcousticFrame.site_id",
    )
    captured_at: datetime = Field(
        description="Timestamp of the SCADA measurement (UTC)"
    )
    pressure_bar: float = Field(description="Line pressure in bar")
    flow_lps: float | None = Field(
        default=None, description="Flow rate in litres per second"
    )
    temperature_c: float | None = Field(
        default=None, description="Water temperature in °C"
    )
    node_ids: list[str] = Field(
        default_factory=list,
        description="OPC-UA NodeIds that were read to produce this sample",
    )
    source: str = Field(
        default="opcua",
        description="'opcua' for real data, 'stub' for simulation",
    )
