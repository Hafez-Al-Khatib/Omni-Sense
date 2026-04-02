"""Pydantic schemas for IEP3 Dispatch Service."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TicketCreateRequest(BaseModel):
    """Payload sent by EEP when a high-confidence leak is detected."""
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    anomaly_score: float
    pipe_material: str
    pressure_bar: float
    scada_mismatch: bool = False


class TicketResponse(BaseModel):
    """Returned immediately after ticket creation."""
    ticket_id: str
    status: Literal["open", "resolved"]
    label: str
    confidence: float
    pipe_material: str
    pressure_bar: float
    scada_mismatch: bool
    created_at: str


class FeedbackRequest(BaseModel):
    """Field technician feedback — closes the active learning loop."""
    ground_truth: str = Field(
        ...,
        description=(
            "True fault class confirmed on-site, or 'false_alarm'. "
            "Must match a known label: Circumferential_Crack, Gasket_Leak, "
            "Longitudinal_Crack, Orifice_Leak, No_Leak, false_alarm."
        ),
    )
    technician_id: str = Field(default="unknown", description="Optional technician identifier.")
    notes: str = Field(default="", description="Optional free-text field notes.")


class FeedbackResponse(BaseModel):
    """Returned after feedback is recorded."""
    ticket_id: str
    status: Literal["resolved"]
    ground_truth: str
    ai_label: str
    correct_prediction: bool
    resolved_at: str


class HealthResponse(BaseModel):
    status: str
    open_tickets: int
    total_tickets: int
