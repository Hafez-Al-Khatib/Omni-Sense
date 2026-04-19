"""Lightweight in-process event bus with a pluggable backend.

In production this is Redpanda/Kafka. For the capstone demo we default to an
in-memory asyncio broker that is API-compatible. Services never call each
other directly — they publish and subscribe by topic.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

log = logging.getLogger("bus")

Handler = Callable[[dict], Awaitable[None]]


# ─────────────────────────── Topics ───────────────────────────────────
class Topics:
    ACOUSTIC_FRAME = "acoustic.frame.v1"
    TELEMETRY = "edge.telemetry.v1"
    DETECTION = "detection.result.v1"
    HYPOTHESIS = "spatial.hypothesis.v1"
    ALERT_NEW = "alert.new.v1"
    ALERT_STATE = "alert.state.v1"
    WORK_ORDER = "workorder.v1"
    NOTIFY = "notify.request.v1"
    AUDIT = "audit.event.v1"
    TWIN_UPDATE = "twin.update.v1"
    SCADA_READING = "scada.reading.v1"


@dataclass
class InMemoryBus:
    _subscribers: dict[str, list[Handler]] = field(default_factory=lambda: defaultdict(list))
    _q: asyncio.Queue = field(default_factory=asyncio.Queue)
    _running: bool = False

    async def publish(self, topic: str, payload: BaseModel | dict) -> None:
        if isinstance(payload, BaseModel):
            payload = json.loads(payload.model_dump_json())
        await self._q.put((topic, payload))

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subscribers[topic].append(handler)
        log.info("subscribed handler to %s (%d total)", topic, len(self._subscribers[topic]))

    async def run(self) -> None:
        self._running = True
        log.info("bus started")
        while self._running:
            topic, payload = await self._q.get()
            handlers = list(self._subscribers.get(topic, ()))
            if not handlers:
                log.debug("no subscribers for %s", topic)
                continue
            # fan-out with isolation: one handler failing must not poison others
            results = await asyncio.gather(
                *(self._safe_call(topic, h, payload) for h in handlers),
                return_exceptions=False,
            )
            del results

    async def _safe_call(self, topic: str, handler: Handler, payload: dict) -> None:
        try:
            await handler(payload)
        except Exception:
            log.exception("handler failed for topic=%s handler=%s", topic, handler.__name__)

    def stop(self) -> None:
        self._running = False


# Module-level singleton for the demo. In production each service owns its own
# Kafka consumer group; this singleton collapses that into one process.
_bus: InMemoryBus | None = None


def get_bus() -> InMemoryBus:
    global _bus
    if _bus is None:
        _bus = InMemoryBus()
    return _bus
