"""
Ticket Store
=============
File-based persistence for maintenance tickets and field feedback.

Tickets are stored as individual JSON files in /tickets/{ticket_id}.json
for simplicity and crash-safety (no partial writes to a single large file).

Feedback is additionally appended to feedback_log.csv — the active learning
input consumed by scripts/train_models.py when retraining is triggered.
"""

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

TICKETS_DIR = Path("tickets")
FEEDBACK_LOG = Path("tickets/feedback_log.csv")

_FEEDBACK_FIELDS = [
    "ticket_id", "created_at", "resolved_at",
    "ai_label", "ground_truth", "correct_prediction",
    "confidence", "pipe_material", "pressure_bar",
    "scada_mismatch", "technician_id", "notes",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ticket_path(ticket_id: str) -> Path:
    return TICKETS_DIR / f"{ticket_id}.json"


def create_ticket(payload: dict) -> dict:
    """Persist a new open ticket and return its full record."""
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)

    ticket_id = str(uuid.uuid4())[:8].upper()  # short human-readable ID
    ticket = {
        "ticket_id":    ticket_id,
        "status":       "open",
        "label":        payload["label"],
        "confidence":   payload["confidence"],
        "probabilities": payload.get("probabilities", {}),
        "anomaly_score": payload.get("anomaly_score", 0.0),
        "pipe_material": payload["pipe_material"],
        "pressure_bar": payload["pressure_bar"],
        "scada_mismatch": payload.get("scada_mismatch", False),
        "created_at":   _now_iso(),
        "resolved_at":  None,
        "ground_truth": None,
        "technician_id": None,
        "notes":        None,
    }

    with open(_ticket_path(ticket_id), "w") as f:
        json.dump(ticket, f, indent=2)

    return ticket


def get_ticket(ticket_id: str) -> dict | None:
    path = _ticket_path(ticket_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_tickets(status: str | None = None) -> list[dict]:
    """Return all tickets, optionally filtered by status."""
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    tickets = []
    for p in sorted(TICKETS_DIR.glob("*.json")):
        with open(p) as f:
            t = json.load(f)
        if status is None or t.get("status") == status:
            tickets.append(t)
    return tickets


def resolve_ticket(ticket_id: str, ground_truth: str, technician_id: str, notes: str) -> dict:
    """
    Mark a ticket as resolved with field-confirmed ground truth.
    Appends a row to feedback_log.csv for the active learning pipeline.
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise KeyError(f"Ticket {ticket_id} not found")
    if ticket["status"] == "resolved":
        raise ValueError(f"Ticket {ticket_id} is already resolved")

    resolved_at = _now_iso()
    ticket.update({
        "status":       "resolved",
        "ground_truth": ground_truth,
        "technician_id": technician_id,
        "notes":        notes,
        "resolved_at":  resolved_at,
    })

    with open(_ticket_path(ticket_id), "w") as f:
        json.dump(ticket, f, indent=2)

    # Append to feedback CSV for active learning
    write_header = not FEEDBACK_LOG.exists()
    with open(FEEDBACK_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FEEDBACK_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "ticket_id":         ticket_id,
            "created_at":        ticket["created_at"],
            "resolved_at":       resolved_at,
            "ai_label":          ticket["label"],
            "ground_truth":      ground_truth,
            "correct_prediction": ticket["label"] == ground_truth,
            "confidence":        ticket["confidence"],
            "pipe_material":     ticket["pipe_material"],
            "pressure_bar":      ticket["pressure_bar"],
            "scada_mismatch":    ticket["scada_mismatch"],
            "technician_id":     technician_id,
            "notes":             notes,
        })

    return ticket
