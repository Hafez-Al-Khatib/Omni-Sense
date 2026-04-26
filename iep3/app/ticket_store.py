"""
Ticket Store (v2.0 - SQLite Backend)
=====================================
Enterprise-grade persistence for maintenance tickets and field feedback.

Optimized for scalability: replaces the O(N) JSON-per-file storage with
an indexed SQLite database. This ensures that listing 10,000+ tickets
takes milliseconds and prevents file-system overhead.

Active Learning:
    Feedback is still appended to feedback_log.csv for the retraining
    pipeline, ensuring backward compatibility with the MLOps flywheel.
"""

import csv
import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("iep3.store")

# --- Paths ---
DB_PATH = Path("tickets/tickets.db")
FEEDBACK_LOG = Path("tickets/feedback_log.csv")
LEGACY_DIR = Path("tickets")

_FEEDBACK_FIELDS = [
    "ticket_id", "created_at", "resolved_at",
    "ai_label", "ground_truth", "correct_prediction",
    "confidence", "pipe_material", "pressure_bar",
    "scada_mismatch", "technician_id", "notes",
]

# --- Database Initialization ---

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return dict-like objects
    return conn

def init_db():
    """Create the tickets table and migrate legacy data if necessary."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                label TEXT,
                confidence REAL,
                probabilities TEXT,
                anomaly_score REAL,
                pipe_material TEXT,
                pressure_bar REAL,
                scada_mismatch INTEGER,
                created_at TEXT,
                resolved_at TEXT,
                ground_truth TEXT,
                technician_id TEXT,
                notes TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON tickets(status)")
        conn.commit()
    
    _migrate_legacy_data()

def _migrate_legacy_data():
    """Finds old .json files and moves them into SQLite."""
    json_files = list(LEGACY_DIR.glob("*.json"))
    if not json_files:
        return

    logger.info(f"Migrating {len(json_files)} legacy JSON tickets to SQLite...")
    with _get_conn() as conn:
        for p in json_files:
            try:
                with open(p) as f:
                    t = json.load(f)
                
                # Check if already exists
                cursor = conn.execute("SELECT 1 FROM tickets WHERE ticket_id = ?", (t["ticket_id"],))
                if cursor.fetchone():
                    p.unlink() # Already migrated
                    continue

                conn.execute("""
                    INSERT INTO tickets (
                        ticket_id, status, label, confidence, probabilities,
                        anomaly_score, pipe_material, pressure_bar, scada_mismatch,
                        created_at, resolved_at, ground_truth, technician_id, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t["ticket_id"], t["status"], t["label"], t.get("confidence"),
                    json.dumps(t.get("probabilities", {})), t.get("anomaly_score", 0.0),
                    t["pipe_material"], t["pressure_bar"], 1 if t.get("scada_mismatch") else 0,
                    t["created_at"], t.get("resolved_at"), t.get("ground_truth"),
                    t.get("technician_id"), t.get("notes")
                ))
                p.unlink() # Delete after successful migration
            except Exception as e:
                logger.error(f"Failed to migrate {p.name}: {e}")
        conn.commit()

# --- Public API ---

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()

def create_ticket(payload: dict) -> dict:
    """Persist a new open ticket in SQLite."""
    init_db() # Ensure DB exists
    ticket_id = str(uuid.uuid4())[:8].upper()
    
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
        "created_at":   _now_iso()
    }

    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO tickets (
                ticket_id, status, label, confidence, probabilities,
                anomaly_score, pipe_material, pressure_bar, scada_mismatch, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id, "open", ticket["label"], ticket["confidence"],
            json.dumps(ticket["probabilities"]), ticket["anomaly_score"],
            ticket["pipe_material"], ticket["pressure_bar"],
            1 if ticket["scada_mismatch"] else 0, ticket["created_at"]
        ))
        conn.commit()
    
    return ticket

def get_ticket(ticket_id: str) -> dict | None:
    init_db()
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        res["probabilities"] = json.loads(res["probabilities"])
        res["scada_mismatch"] = bool(res["scada_mismatch"])
        return res

def list_tickets(status: str | None = None) -> list[dict]:
    """Return all tickets from DB, filtered by status if provided."""
    init_db()
    query = "SELECT * FROM tickets"
    params = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    
    query += " ORDER BY created_at DESC"
    
    tickets = []
    with _get_conn() as conn:
        for row in conn.execute(query, params):
            t = dict(row)
            t["probabilities"] = json.loads(t["probabilities"])
            t["scada_mismatch"] = bool(t["scada_mismatch"])
            tickets.append(t)
    return tickets

def resolve_ticket(ticket_id: str, ground_truth: str, technician_id: str, notes: str) -> dict:
    """Resolve a ticket and update both SQLite and the CSV feedback log."""
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise KeyError(f"Ticket {ticket_id} not found")
    if ticket["status"] == "resolved":
        raise ValueError(f"Ticket {ticket_id} is already resolved")

    resolved_at = _now_iso()
    correct = (ticket["label"] == ground_truth)

    with _get_conn() as conn:
        conn.execute("""
            UPDATE tickets SET
                status = 'resolved',
                ground_truth = ?,
                technician_id = ?,
                notes = ?,
                resolved_at = ?
            WHERE ticket_id = ?
        """, (ground_truth, technician_id, notes, resolved_at, ticket_id))
        conn.commit()

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
            "correct_prediction": correct,
            "confidence":        ticket["confidence"],
            "pipe_material":     ticket["pipe_material"],
            "pressure_bar":      ticket["pressure_bar"],
            "scada_mismatch":    ticket["scada_mismatch"],
            "technician_id":     technician_id,
            "notes":             notes,
        })

    return get_ticket(ticket_id)
