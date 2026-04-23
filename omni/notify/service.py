"""Notification service — public interface.

Consumes notify.request.v1 and routes to the correct channel. In production
with Twilio credentials set, real SMS (and WhatsApp for CRITICAL) messages are
dispatched to field crews. Without credentials the service logs to an in-memory
inbox for the ops console.

This module is the stable public API: existing callers of ``notify_service.wire()``
and ``notify_service.INBOX`` continue to work unchanged.  The actual delivery
logic lives in twilio_service.py, which this module delegates to.

Channels (in priority order)
-----------------------------
TWILIO   — real SMS / WhatsApp via twilio_service.on_notify()
STUB     — log-only fallback when Twilio env vars are absent

Selection happens automatically at import time based on env vars.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime

from omni.common.bus import Topics, get_bus
from omni.notify import twilio_service

log = logging.getLogger("notify")

# In-memory inbox visible to the ops console (always populated regardless of
# which delivery backend is active).
INBOX: deque = deque(maxlen=500)


async def on_notify(payload: dict) -> None:
    """Stamp the payload, append to INBOX, then delegate to twilio_service.

    INBOX is always populated so the ops console can display recent
    notifications even when Twilio is not configured.
    """
    payload["sent_at"] = datetime.now(UTC).isoformat()
    INBOX.append(payload)
    log.info(
        "notify → [%s] %s",
        payload.get("severity"),
        payload.get("subject"),
    )
    # Delegate actual delivery (SMS / WhatsApp / stub) to twilio_service.
    # twilio_service.on_notify() handles Twilio-configured vs stub-mode
    # internally; we never need to branch here.
    await twilio_service.on_notify(payload)


def wire() -> None:
    """Subscribe on_notify to Topics.NOTIFY.

    A single handler covers both INBOX bookkeeping and Twilio delivery.
    Twilio is used automatically when credentials are present in env;
    otherwise the module falls back to log-only stub mode.
    """
    twilio_service._log_startup()  # emit Twilio-configured / stub-mode log line
    get_bus().subscribe(Topics.NOTIFY, on_notify)
