"""Pydantic schemas for EEP."""

from pydantic import BaseModel, Field


class MetadataInput(BaseModel):
    """Metadata about the recording environment."""
    pipe_material: str = Field(
        default="PVC",
        pattern="^(PVC|Steel|Cast_Iron)$",
        description="Pipe material type",
    )
    pressure_bar: float = Field(
        default=3.0,
        ge=0.1,
        le=20.0,
        description="Pipe pressure in bar",
    )


class DiagnoseResult(BaseModel):
    """Full diagnosis response from the pipeline."""
    label: str
    confidence: float
    probabilities: dict[str, float]
    anomaly_score: float
    is_in_distribution: bool
    signal_quality: dict
    scada_mismatch: bool = False
    scada_mismatch_detail: str | None = None
    baseline_decision: str = "unknown"
    baseline_rms: float = 0.0


class OODResult(BaseModel):
    """Response when Out-of-Distribution is detected."""
    error: str
    detail: str
    anomaly_score: float
    threshold: float
    signal_quality: dict


class CalibrateResult(BaseModel):
    """Response from the calibration endpoint."""
    message: str
    num_samples: int
    new_threshold: float
    ambient_score_mean: float
    ambient_score_std: float
