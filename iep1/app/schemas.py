"""Pydantic schemas for IEP1."""

from pydantic import BaseModel


class EmbeddingResponse(BaseModel):
    """Response from the /embed endpoint."""
    embedding: list[float]
    embedding_dim: int
    duration_samples: int
    sample_rate: int


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    model_loaded: bool
