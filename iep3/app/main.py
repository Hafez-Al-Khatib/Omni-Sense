"""
IEP 3 — Dispatch & Active Learning Service
============================================
Closes the MLOps flywheel by converting high-confidence AI predictions
into actionable maintenance tickets and routing field-technician feedback
back into the training pipeline.

Lifecycle:
    EEP detects high-confidence leak (>90%)
        → POST /api/v1/ticket            (EEP, fire-and-forget)
        → Ticket created, UUID assigned

    Field technician receives alert
        → POST /api/v1/feedback/{id}     (mobile/web app)
        → Ground-truth label recorded
        → feedback_log.csv updated       (active learning input)

    Retraining pipeline reads feedback_log.csv
        → New labelled samples flow into next training run
        → Model improves from real-world corrections

Endpoints:
    POST  /api/v1/ticket              — Create maintenance ticket
    POST  /api/v1/feedback/{id}       — Record technician feedback
    GET   /api/v1/tickets             — List all tickets (optional ?status=open)
    GET   /api/v1/tickets/{id}        — Get single ticket
    GET   /health                     — Health check
"""

import logging
import os

from fastapi import FastAPI, HTTPException, Query, Request
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import ticket_store
from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    TicketCreateRequest,
    TicketResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iep3")

# ─── Prometheus ───────────────────────────────────────────────────────────────

TICKETS_CREATED = Counter("iep3_tickets_created_total", "Maintenance tickets created")
FEEDBACK_RECEIVED = Counter(
    "iep3_feedback_received_total",
    "Technician feedback received",
    ["correct"],  # label: "true" | "false"
)
OPEN_TICKETS = Gauge("iep3_open_tickets", "Number of unresolved maintenance tickets")

# ─── App ──────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Omni-Sense IEP3 — Dispatch & Active Learning",
    description="Converts AI predictions into maintenance tickets; routes feedback to MLflow.",
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

Instrumentator().instrument(app).expose(app)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    all_tickets = ticket_store.list_tickets()
    open_tickets = [t for t in all_tickets if t["status"] == "open"]
    return HealthResponse(
        status="healthy",
        open_tickets=len(open_tickets),
        total_tickets=len(all_tickets),
    )


@app.post("/api/v1/ticket", response_model=TicketResponse, status_code=201)
@limiter.limit(os.getenv("OMNI_IEP_RATE_LIMIT", "100/minute"))
def create_ticket(request: Request, request_body: TicketCreateRequest):
    """
    Create a maintenance ticket from a high-confidence AI prediction.
    Called by EEP as a fire-and-forget background task.
    """
    ticket = ticket_store.create_ticket(request_body.model_dump())
    TICKETS_CREATED.inc()
    OPEN_TICKETS.inc()
    logger.info(
        f"Ticket {ticket['ticket_id']} created: "
        f"label={ticket['label']} confidence={ticket['confidence']:.2f}"
    )
    return TicketResponse(**ticket)


@app.post("/api/v1/feedback/{ticket_id}", response_model=FeedbackResponse)
@limiter.limit(os.getenv("OMNI_IEP_RATE_LIMIT", "100/minute"))
def submit_feedback(request: Request, ticket_id: str, request_body: FeedbackRequest):
    """
    Record field-technician ground-truth feedback for a ticket.

    This is the active learning entry point — each submission appends
    a verified label to feedback_log.csv, which the retraining pipeline
    picks up on its next run.
    """
    try:
        ticket = ticket_store.resolve_ticket(
            ticket_id=ticket_id,
            ground_truth=request_body.ground_truth,
            technician_id=request_body.technician_id,
            notes=request_body.notes,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    correct = ticket["label"] == request.ground_truth
    FEEDBACK_RECEIVED.labels(correct=str(correct).lower()).inc()
    OPEN_TICKETS.dec()

    logger.info(
        f"Feedback for {ticket_id}: ai={ticket['label']} "
        f"ground_truth={request.ground_truth} correct={correct}"
    )

    return FeedbackResponse(
        ticket_id=ticket_id,
        status="resolved",
        ground_truth=request.ground_truth,
        ai_label=ticket["label"],
        correct_prediction=correct,
        resolved_at=ticket["resolved_at"],
    )


@app.get("/api/v1/tickets", response_model=list[TicketResponse])
def list_tickets(status: str | None = Query(default=None, pattern="^(open|resolved)$")):
    """List all tickets, optionally filtered by ?status=open or ?status=resolved."""
    return [TicketResponse(**t) for t in ticket_store.list_tickets(status=status)]


@app.get("/api/v1/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: str):
    """Retrieve a single ticket by ID."""
    ticket = ticket_store.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return TicketResponse(**ticket)
