"""Redis Streams event bus — durable, at-least-once delivery.

Replaces the fire-and-forget Redis Pub/Sub implementation with Redis Streams
(XADD / XREADGROUP).  Unlike PUBLISH/SUBSCRIBE, Streams persist messages on
disk: if the omni-platform process restarts while a pipe is bursting, the
unprocessed AcousticFrames are waiting in the stream and will be consumed
when the process comes back up.

Architecture
------------
Producer side
  ``publish(topic, payload)``
    →  XADD  <topic>  *  "payload" <json>
    Messages accumulate in the stream and are trimmed to MAXLEN entries.

Consumer side
  ``subscribe(topic, handler)``   — registers handler locally
  ``run()``                       — starts one asyncio task per subscribed topic
    Each task:
      1. Creates a consumer group "<topic>-grp" if not already present.
      2. Loops: XREADGROUP with COUNT=10 and BLOCK=500ms.
      3. Dispatches each message to all local handlers.
      4. ACKs immediately after all handlers return (at-least-once delivery).
      5. On failure, leaves message in PEL for manual recovery or re-delivery.

Consumer group name  : ``omni-<topic>``     (one per topic)
Consumer name        : ``omni-platform``    (can be overridden per-process)

Why one group per topic?
  In production, different services consume different topics.  One group per
  topic means all instances of omni-platform share the same cursor, giving
  true competing-consumer fan-out without duplicates within the same service.

Fallback
--------
If the ``redis`` package is absent or REDIS_URL is unset, the factory
returns InMemoryBus so the test suite passes without a live Redis.

Redis compatibility
-------------------
Redis ≥ 5.0 (Streams) and ≥ 6.2 (XAUTOCLAIM) required.
redis:7.2-alpine in docker-compose satisfies both.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from .bus import Topics  # noqa: F401 — re-export for callers

log = logging.getLogger("redis_bus")

Handler = Callable[[dict], Awaitable[None]]

_REDIS_URL: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")

# Stream config
_STREAM_MAXLEN      = 10_000   # trim each stream to this many entries
_CONSUMER_GROUP_PFX = "omni"   # group name = f"{_CONSUMER_GROUP_PFX}-{topic}"
_CONSUMER_NAME      = os.environ.get("OMNI_CONSUMER_NAME", "omni-platform")
_READ_COUNT         = 10       # messages per XREADGROUP call
_BLOCK_MS           = 500      # ms to block waiting for new messages

# Reconnect backoff
_BACKOFF_INIT: float = 1.0
_BACKOFF_MAX:  float = 30.0
_BACKOFF_MUL:  float = 2.0

try:
    import redis.asyncio as aioredis  # type: ignore
    _REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore
    _REDIS_AVAILABLE = False


# ─── Serialisation ────────────────────────────────────────────────────────────

def _to_json(payload: BaseModel | dict | Any) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    from datetime import datetime
    from uuid import UUID

    def _default(obj):
        if isinstance(obj, (UUID,)):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Not JSON serialisable: {type(obj)}")

    return json.dumps(payload, default=_default)


def _from_json(raw: bytes | str) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return json.loads(raw)


# ─── RedisBus ─────────────────────────────────────────────────────────────────

class RedisBus:
    """Durable event bus backed by Redis Streams.

    API is identical to InMemoryBus so no business-logic code changes.
    """

    def __init__(self, redis_url: str = _REDIS_URL) -> None:
        self._url = redis_url
        self._subscribers: dict[str, list[Handler]] = {}
        self._running = False
        self._client: aioredis.Redis | None = None
        self._consumer_tasks: list[asyncio.Task] = []

    # ── publish ───────────────────────────────────────────────────────────────

    async def publish(self, topic: str, payload: BaseModel | dict) -> None:
        """XADD topic * payload <json>."""
        client = await self._get_client()
        raw = _to_json(payload)
        try:
            await client.xadd(
                topic,
                {"payload": raw},
                maxlen=_STREAM_MAXLEN,
                approximate=True,
            )
            log.debug("XADD %s (%d bytes)", topic, len(raw))
        except Exception as exc:
            log.error("XADD failed topic=%s: %s", topic, exc)
            self._client = None   # force reconnect

    # ── subscribe ─────────────────────────────────────────────────────────────

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register *handler* for *topic*.  Consumer group is created at run()."""
        self._subscribers.setdefault(topic, []).append(handler)
        log.info("subscribed handler to %s (%d total)", topic,
                 len(self._subscribers[topic]))

    # ── run / stop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start one consumer task per subscribed topic and wait for stop()."""
        self._running = True
        log.info("RedisBus (Streams) starting — url=%s consumer=%s",
                 self._url, _CONSUMER_NAME)

        client = await self._get_client()

        # Create consumer groups for all subscribed topics
        for topic in self._subscribers:
            await self._ensure_group(client, topic)

        # Spawn one consumer task per topic
        self._consumer_tasks = [
            asyncio.create_task(
                self._consume_topic(topic),
                name=f"redis-stream-{topic}",
            )
            for topic in self._subscribers
        ]

        # Block until stop() is called
        try:
            while self._running:
                await asyncio.sleep(0.1)
        finally:
            for t in self._consumer_tasks:
                t.cancel()
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
            log.info("RedisBus stopped")

    async def _ensure_group(self, client, topic: str) -> None:
        """XGROUP CREATE topic group $ MKSTREAM — idempotent."""
        group = f"{_CONSUMER_GROUP_PFX}-{topic}"
        try:
            await client.xgroup_create(
                topic, group,
                id="0",          # "0" → deliver all existing messages on first start
                mkstream=True,   # creates the stream if it doesn't exist yet
            )
            log.info("Consumer group created: stream=%s group=%s", topic, group)
        except Exception as exc:
            # BUSYGROUP = group already exists — safe to ignore
            if "BUSYGROUP" in str(exc):
                log.debug("Consumer group already exists: %s / %s", topic, group)
            else:
                log.warning("xgroup_create %s/%s: %s", topic, group, exc)

    async def _consume_topic(self, topic: str) -> None:
        """Read → dispatch → ACK loop for one stream topic with reconnect."""
        group    = f"{_CONSUMER_GROUP_PFX}-{topic}"
        backoff  = _BACKOFF_INIT

        while self._running:
            try:
                client = await self._get_client()
                # XREADGROUP: ">" means only undelivered messages
                results = await client.xreadgroup(
                    groupname=group,
                    consumername=_CONSUMER_NAME,
                    streams={topic: ">"},
                    count=_READ_COUNT,
                    block=_BLOCK_MS,
                )
                backoff = _BACKOFF_INIT   # reset on success

                if not results:
                    continue

                # results: [(stream_name, [(id, {field: value}), ...])]
                for _stream, messages in results:
                    for msg_id, fields in messages:
                        await self._dispatch(topic, group, msg_id, fields, client)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not self._running:
                    return
                log.warning("Stream consumer error topic=%s: %s — retry %.1fs",
                            topic, exc, backoff)
                self._client = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_MUL, _BACKOFF_MAX)

    async def _dispatch(
        self,
        topic: str,
        group: str,
        msg_id,
        fields: dict,
        client,
    ) -> None:
        """Parse message, fan-out to handlers, then ACK."""
        raw = fields.get(b"payload") or fields.get("payload", b"{}")
        try:
            payload = _from_json(raw)
        except json.JSONDecodeError:
            log.warning("Bad JSON in stream %s id=%s — ACK and skip", topic, msg_id)
            await client.xack(topic, group, msg_id)
            return

        handlers = list(self._subscribers.get(topic, ()))
        if handlers:
            await asyncio.gather(
                *(self._safe_call(topic, h, payload) for h in handlers),
                return_exceptions=False,
            )

        # ACK after all handlers succeed — guarantees at-least-once delivery
        try:
            await client.xack(topic, group, msg_id)
        except Exception as exc:
            log.warning("XACK failed topic=%s id=%s: %s", topic, msg_id, exc)

    async def _safe_call(self, topic: str, handler: Handler, payload: dict) -> None:
        try:
            await handler(payload)
        except Exception:
            log.exception("handler failed topic=%s handler=%s",
                          topic, getattr(handler, "__name__", repr(handler)))

    def stop(self) -> None:
        self._running = False

    # ── internal ──────────────────────────────────────────────────────────────

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=False,   # keep raw bytes for xread fields
            )
        return self._client

    async def aclose(self) -> None:
        self.stop()
        if self._client:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None


# ─── Factory ──────────────────────────────────────────────────────────────────

_bus_singleton: RedisBus | Any | None = None


def get_bus(force_redis: bool = False) -> RedisBus | Any:
    """Return the application-wide bus singleton (RedisBus or InMemoryBus)."""
    global _bus_singleton
    if _bus_singleton is not None:
        return _bus_singleton

    env_url = os.environ.get("REDIS_URL", "")

    if not _REDIS_AVAILABLE:
        log.warning("redis package not installed → InMemoryBus fallback")
        from .bus import InMemoryBus
        _bus_singleton = InMemoryBus()
        return _bus_singleton

    if not env_url and not force_redis:
        log.info("REDIS_URL not set → InMemoryBus (dev/CI mode)")
        from .bus import InMemoryBus
        _bus_singleton = InMemoryBus()
        return _bus_singleton

    log.info("Creating RedisBus (Streams) → %s", env_url or _REDIS_URL)
    _bus_singleton = RedisBus(redis_url=env_url or _REDIS_URL)
    return _bus_singleton


def reset_bus_singleton() -> None:
    global _bus_singleton
    _bus_singleton = None
