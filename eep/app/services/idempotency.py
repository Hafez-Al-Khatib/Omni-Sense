"""
Idempotency Service
====================
Prevents duplicate processing of diagnostic requests on unstable networks.
Caches results in Redis using the X-Idempotency-Key header.
"""

import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis
from app.config import settings

logger = logging.getLogger("eep.idempotency")

# Cache results for 5 minutes (standard window for mobile app retries)
_CACHE_TTL_SECONDS = 300

class IdempotencyService:
    def __init__(self, redis_url: str):
        self._url = redis_url
        self._client: Optional[aioredis.Redis] = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._url, encoding="utf-8", decode_responses=True)
        return self._client

    async def get_cached_result(self, key: str) -> Optional[dict[str, Any]]:
        """Retrieve a cached result if it exists."""
        if not key:
            return None
        
        try:
            client = await self._get_client()
            raw = await client.get(f"idempotency:{key}")
            if raw:
                logger.info(f"Idempotency HIT for key: {key}")
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"Idempotency cache read failed: {e}")
        
        return None

    async def save_result(self, key: str, result: dict[str, Any]) -> None:
        """Cache a successful result."""
        if not key:
            return
        
        try:
            client = await self._get_client()
            await client.setex(
                f"idempotency:{key}",
                _CACHE_TTL_SECONDS,
                json.dumps(result)
            )
            logger.debug(f"Idempotency stored for key: {key}")
        except Exception as e:
            logger.warning(f"Idempotency cache write failed: {e}")

# Singleton
idempotency_service = IdempotencyService(settings.REDIS_URL)
