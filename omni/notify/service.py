"""Notification service.

Consumes notify.request.v1 and routes to the correct channel. In production:
Twilio (SMS), FCM (push), SendGrid (email), PagerDuty (paging). Here we
append to an in-memory inbox and log — but the interface is identical.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone

from omni.common.bus import Topics, get_bus

log = logging.getLogger("notify")

INBOX: deque = deque(maxlen=500)


async def on_notify(payload: dict) -> None:
    payload["sent_at"] = datetime.now(timezone.utc).isoformat()
    INBOX.append(payload)
    log.info(
        "notify → %s [%s] %s",
        payload.get("channel"),
        payload.get("severity"),
        payload.get("subject"),
    )


def wire() -> None:
    get_bus().subscribe(Topics.NOTIFY, on_notify)
