"""Shared Pydantic schemas for Omni-Sense platform v2.

Every event on the bus validates against one of these. Versioned explicitly
so we can evolve without breaking consumers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
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
class Severity(str, Enum):
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
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Per-head outputs
    yamnet_top_class: Optional[str] = None
    xgb_p_leak: float = Field(ge=0.0, le=1.0)
    rf_p_leak: float = Field(ge=0.0, le=1.0)
    cnn_p_leak: Optional[float] = None
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    contributing_detection_ids: list[UUID]
    lat: float
    lon: float
    uncertainty_m: float = Field(description="95% confidence radius in meters")
    pipe_segment_id: Optional[str] = None
    distance_along_pipe_m: Optional[float] = None
    estimated_flow_lps: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)


# ─────────────────────────── Alerts ───────────────────────────────────
class AlertState(str, Enum):
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: AlertState = AlertState.NEW
    severity: Severity
    severity_score: float = Field(ge=0.0, le=100.0)
    title: str
    summary: str
    lat: float
    lon: float
    pipe_segment_id: Optional[str] = None
    estimated_loss_lph: Optional[float] = None
    assigned_crew_id: Optional[str] = None
    sla_due_at: Optional[datetime] = None
    history: list[dict] = Field(default_factory=list)


# ─────────────────────────── Work orders ──────────────────────────────
class WorkOrderStatus(str, Enum):
    DRAFT = "DRAFT"
    DISPATCHED = "DISPATCHED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class WorkOrder(BaseModel):
    schema_version: Literal["1"] = "1"
    work_order_id: UUID = Field(default_factory=uuid4)
    alert_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: WorkOrderStatus = WorkOrderStatus.DRAFT
    crew_id: str
    eta_minutes: int
    parts_required: list[str] = Field(default_factory=list)
    notes: str = ""
    completed_at: Optional[datetime] = None
    repair_cost_usd: Optional[float] = None
    mtbf_days: Optional[float] = None


# ─────────────────────────── Audit ────────────────────────────────────
class AuditEvent(BaseModel):
    schema_version: Literal["1"] = "1"
    event_id: UUID = Field(default_factory=uuid4)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str
    action: str
    resource_type: str
    resource_id: str
    payload_hash_sha256: str
    prev_hash: str
    signature_ed25519: str
