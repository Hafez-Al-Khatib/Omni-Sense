"""Pydantic schemas for IEP2."""

from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    """Request to the /diagnose endpoint."""
    embedding: list[float] = Field(..., min_length=1, max_length=2048)
    pipe_material: str = Field(default="PVC", pattern="^(PVC|Steel|Cast_Iron)$")
    pressure_bar: float = Field(default=3.0, ge=0.1, le=20.0)


class DiagnoseResponse(BaseModel):
    """Successful diagnosis response."""
    label: str
    confidence: float
    probabilities: dict[str, float]
    anomaly_score: float
    is_in_distribution: bool
    scada_mismatch: bool = False
    scada_mismatch_detail: str | None = None


class CalibrateRequest(BaseModel):
    """Request to the /calibrate endpoint."""
    ambient_embeddings: list[list[float]] = Field(
        ...,
        min_length=1,
        description="List of 1024-d ambient embeddings for calibration",
    )


class CalibrateResponse(BaseModel):
    """Response from the /calibrate endpoint."""
    message: str
    num_samples: int
    new_threshold: float
    ambient_score_mean: float
    ambient_score_std: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ood_model_loaded: bool
    classifier_loaded: bool
