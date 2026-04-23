"""Twilio SMS / WhatsApp notification service for Omni-Sense.

Replaces the stub notify service when Twilio credentials are present in the
environment.  Uses only Python stdlib (urllib) — no Twilio SDK required.

Environment variables
---------------------
TWILIO_ACCOUNT_SID      Twilio account SID (ACxxx...)
TWILIO_AUTH_TOKEN       Twilio auth token
TWILIO_FROM_NUMBER      E.164 SMS originator, e.g. +12025551234
TWILIO_CREW_NUMBERS     JSON map: { "crew-id": "+9613XXXXXX", ... }

When TWILIO_ACCOUNT_SID is absent or empty the module silently falls back to
the stub (log-only) implementation, so local development requires no config.

Behaviour by severity
---------------------
HIGH      → SMS to the assigned crew's phone number
CRITICAL  → SMS **and** WhatsApp to the assigned crew's phone number

Prometheus counters
-------------------
omni_sms_sent_total{status="success"|"failed", severity="high"|"critical"}
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import UTC, datetime

try:
    from prometheus_client import Counter
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

from omni.common.bus import Topics, get_bus

log = logging.getLogger("notify.twilio")

# ─────────────────────── Prometheus metrics ────────────────────────────────────

if _HAS_PROMETHEUS:
    SMS_COUNTER = Counter(
        "omni_sms_sent_total",
        "Total SMS/WhatsApp messages sent via Twilio",
        labelnames=["status", "severity"],
    )
else:
    class _FakeCounter:  # pragma: no cover
        def labels(self, **_): return self
        def inc(self): pass
    SMS_COUNTER = _FakeCounter()  # type: ignore[assignment]

# ─────────────────────── Delivery receipt log ──────────────────────────────────

SENT_LOG: deque = deque(maxlen=500)

# ─────────────────────── Config ────────────────────────────────────────────────

_ACCOUNT_SID: str = os.environ.get("TWILIO_ACCOUNT_SID", "")
_AUTH_TOKEN: str = os.environ.get("TWILIO_AUTH_TOKEN", "")
_FROM_NUMBER: str = os.environ.get("TWILIO_FROM_NUMBER", "")

def _crew_numbers() -> dict[str, str]:
    """Parse TWILIO_CREW_NUMBERS JSON env var; return empty dict on error."""
    raw = os.environ.get("TWILIO_CREW_NUMBERS", "")
    if not raw:
        return {}
    try:
        mapping = json.loads(raw)
        if isinstance(mapping, dict):
            return {str(k): str(v) for k, v in mapping.items()}
    except (json.JSONDecodeError, TypeError):
        log.warning("TWILIO_CREW_NUMBERS is not valid JSON — crew SMS disabled")
    return {}


def _twilio_configured() -> bool:
    """Return True only when all required Twilio credentials are present."""
    return bool(_ACCOUNT_SID and _AUTH_TOKEN and _FROM_NUMBER)


# ─────────────────────── Message formatting ────────────────────────────────────

def _build_message(payload: dict) -> str:
    """Format the SMS / WhatsApp message body from a notify payload."""
    severity = payload.get("severity", "UNKNOWN").upper()
    subject = payload.get("subject", "(no subject)")
    pipe = payload.get("pipe_segment_id") or "N/A"
    crew = payload.get("crew_id") or "N/A"
    loss = payload.get("estimated_loss_lph")
    lat = payload.get("lat")
    lon = payload.get("lon")

    loss_str = f"{loss:.0f}" if loss is not None else "N/A"
    loc_str = f"{lat:.5f},{lon:.5f}" if lat is not None and lon is not None else "N/A"

    return (
        f"[OMNI-SENSE {severity}] {subject}\n"
        f"Pipe: {pipe}\n"
        f"Crew: {crew}\n"
        f"Loss: {loss_str} L/h\n"
        f"Location: {loc_str}"
    )


# ─────────────────────── Twilio REST API call ──────────────────────────────────

def _post_message_sync(to: str, body: str) -> dict:
    """Synchronous Twilio Messages.json POST.  Called via asyncio.to_thread."""
    url = (
        f"https://api.twilio.com/2010-04-01/Accounts/{_ACCOUNT_SID}/Messages.json"
    )
    credentials = base64.b64encode(
        f"{_ACCOUNT_SID}:{_AUTH_TOKEN}".encode()
    ).decode()
    data = urllib.parse.urlencode({
        "To": to,
        "From": _FROM_NUMBER,
        "Body": body,
    }).encode()

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "omni-sense/2.0 (urllib)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            response_body = resp.read()
            result = json.loads(response_body)
            return {"status": "success", "sid": result.get("sid"), "to": to}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        log.error("Twilio HTTP %d for %s: %s", exc.code, to, error_body)
        return {"status": "failed", "error": f"HTTP {exc.code}", "to": to}
    except urllib.error.URLError as exc:
        log.error("Twilio network error for %s: %s", to, exc.reason)
        return {"status": "failed", "error": str(exc.reason), "to": to}
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected Twilio error for %s", to)
        return {"status": "failed", "error": str(exc), "to": to}


async def _send_sms(to: str, body: str, severity: str) -> None:
    """Send a plain SMS and record the outcome."""
    log.info("Twilio SMS → %s [%s]", to, severity.upper())
    result = await asyncio.to_thread(_post_message_sync, to, body)
    result["channel"] = "sms"
    result["severity"] = severity
    result["sent_at"] = datetime.now(UTC).isoformat()
    SENT_LOG.append(result)
    SMS_COUNTER.labels(status=result["status"], severity=severity.lower()).inc()
    if result["status"] == "success":
        log.info("SMS delivered  sid=%s to=%s", result.get("sid"), to)
    else:
        log.warning("SMS failed     error=%s to=%s", result.get("error"), to)


async def _send_whatsapp(to_number: str, body: str, severity: str) -> None:
    """Send a WhatsApp message (Twilio WhatsApp sandbox / approved number).

    Twilio WhatsApp uses the same Messages.json endpoint but prefixes the
    recipient with 'whatsapp:'.
    """
    whatsapp_to = f"whatsapp:{to_number}"
    log.info("Twilio WhatsApp → %s [%s]", whatsapp_to, severity.upper())
    result = await asyncio.to_thread(_post_message_sync, whatsapp_to, body)
    result["channel"] = "whatsapp"
    result["severity"] = severity
    result["sent_at"] = datetime.now(UTC).isoformat()
    SENT_LOG.append(result)
    SMS_COUNTER.labels(status=result["status"], severity=severity.lower()).inc()
    if result["status"] == "success":
        log.info("WhatsApp delivered  sid=%s to=%s", result.get("sid"), whatsapp_to)
    else:
        log.warning("WhatsApp failed     error=%s to=%s", result.get("error"), whatsapp_to)


# ─────────────────────── Main handler ─────────────────────────────────────────

async def on_notify(payload: dict) -> None:
    """Handle a notify.request.v1 payload.

    Routes to Twilio for HIGH/CRITICAL severity when credentials are configured;
    falls back to stub (log-only) otherwise.
    """
    severity: str = (payload.get("severity") or "info").lower()
    crew_id: str | None = payload.get("crew_id")

    # Always stamp and log — this matches the stub's behaviour so INBOX
    # (in service.py) still fills for the ops console even in Twilio mode.
    payload.setdefault("sent_at", datetime.now(UTC).isoformat())
    log.info(
        "notify → %s [%s] %s",
        payload.get("channel", "twilio"),
        severity,
        payload.get("subject"),
    )

    if severity not in ("high", "critical"):
        # INFO / LOW / MEDIUM — log only, no external message
        return

    if not _twilio_configured():
        log.warning(
            "Twilio not configured — skipping SMS for severity=%s (set "
            "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER)",
            severity,
        )
        return

    crew_map = _crew_numbers()
    to_number: str | None = crew_map.get(crew_id) if crew_id else None

    if not to_number:
        log.warning(
            "No phone number for crew_id=%s — cannot send SMS (check TWILIO_CREW_NUMBERS)",
            crew_id,
        )
        return

    body = _build_message(payload)

    # Always send SMS for HIGH and CRITICAL
    await _send_sms(to_number, body, severity)

    # For CRITICAL also attempt WhatsApp
    if severity == "critical":
        try:
            await _send_whatsapp(to_number, body, severity)
        except Exception:  # noqa: BLE001
            # WhatsApp failure must never prevent the rest of the pipeline
            log.exception("WhatsApp delivery failed — SMS was already sent")


# ─────────────────────── Wire / startup helpers ───────────────────────────────

def _log_startup() -> None:
    """Emit a one-line startup log indicating whether Twilio is active."""
    if _twilio_configured():
        log.info(
            "Twilio notify service active (SID=%s...%s, from=%s)",
            _ACCOUNT_SID[:4],
            _ACCOUNT_SID[-4:],
            _FROM_NUMBER,
        )
    else:
        log.info(
            "Twilio credentials not set — running in stub mode "
            "(HIGH/CRITICAL alerts will be logged only)"
        )


def wire() -> None:
    """Subscribe on_notify directly to Topics.NOTIFY.

    Use this only when running twilio_service standalone (e.g. in tests).
    In normal platform operation, omni/notify/service.py calls wire() which
    registers its own wrapper around on_notify; twilio_service.wire() would
    then create a second, duplicate subscription.  Prefer service.wire().
    """
    _log_startup()
    get_bus().subscribe(Topics.NOTIFY, on_notify)
