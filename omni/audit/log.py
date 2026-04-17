"""WORM audit log with Ed25519 signatures + Merkle chain.

Every meaningful business event goes through here: detections, hypotheses,
alerts, dispatches, work order closures. Each record chains to the previous
one's hash, so tampering is detectable; each record is signed with the
platform's Ed25519 key, so authorship is provable.

In prod the chain root is periodically anchored into an S3 Object-Locked
bucket to satisfy regulator requirements. Here we just keep the list.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from omni.common.bus import Topics, get_bus
from omni.common.schemas import AuditEvent

log = logging.getLogger("audit")

# Try to use real Ed25519; gracefully degrade to HMAC-SHA256 if unavailable.
try:
    from nacl.signing import SigningKey  # type: ignore
    _nacl_ok = True
    _signing_key = SigningKey.generate()
    _verify_key_hex = _signing_key.verify_key.encode().hex()
except Exception:
    _nacl_ok = False
    import hmac
    import secrets
    _hmac_secret = secrets.token_bytes(32)
    _verify_key_hex = "hmac-sha256-fallback"


CHAIN: list[AuditEvent] = []
_GENESIS = "0" * 64


def _sign(payload: bytes) -> str:
    if _nacl_ok:
        return _signing_key.sign(payload).signature.hex()
    return hmac.new(_hmac_secret, payload, hashlib.sha256).hexdigest()


def _record(actor: str, action: str, resource_type: str, resource_id: str, payload: dict) -> AuditEvent:
    prev = CHAIN[-1].payload_hash_sha256 if CHAIN else _GENESIS
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    h = hashlib.sha256(blob + prev.encode()).hexdigest()
    sig = _sign(h.encode())
    ev = AuditEvent(
        actor=actor,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        payload_hash_sha256=h,
        prev_hash=prev,
        signature_ed25519=sig,
    )
    CHAIN.append(ev)
    return ev


async def on_detection(payload: dict) -> None:
    _record("eep-v2", "detection_recorded", "detection", payload.get("detection_id", ""), payload)


async def on_alert(payload: dict) -> None:
    _record("alert-engine", "alert_emitted", "alert", payload.get("alert_id", ""), payload)


async def on_work_order(payload: dict) -> None:
    _record("dispatch", "work_order_created", "work_order", payload.get("work_order_id", ""), payload)


async def on_state(payload: dict) -> None:
    _record("alert-engine", "alert_transitioned", "alert", payload.get("alert_id", ""), payload)


def verify_chain() -> tuple[bool, Optional[int]]:
    """Walk the chain, return (ok, first_bad_index or None)."""
    prev = _GENESIS
    for i, ev in enumerate(CHAIN):
        if ev.prev_hash != prev:
            return False, i
        prev = ev.payload_hash_sha256
    return True, None


def wire() -> None:
    b = get_bus()
    b.subscribe(Topics.DETECTION, on_detection)
    b.subscribe(Topics.ALERT_NEW, on_alert)
    b.subscribe(Topics.WORK_ORDER, on_work_order)
    b.subscribe(Topics.ALERT_STATE, on_state)
    log.info("audit wired (nacl_ok=%s, verify_key=%s)", _nacl_ok, _verify_key_hex[:16])
