"""Pydantic schemas for IEP4."""

from pydantic import BaseModel


class CNNResponse(BaseModel):
    """Successful CNN classification response."""
    label: str
    confidence: float
    probabilities: dict[str, float]
    backend: str
    model_loaded: bool
    # Autoencoder OOD fields (optional — absent until autoencoder is trained)
    ood_reconstruction_error: float | None = None
    is_ood: bool = False
    ood_threshold: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str | None = None
    autoencoder_loaded: bool = False
