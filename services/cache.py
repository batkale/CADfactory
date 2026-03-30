"""
Optional Redis caching layer.

If Redis is not configured or unavailable, all cache operations are no-ops
so the app continues to work without caching.

Usage:
    from services.cache import cache_get, cache_set, cache_delete, make_key

    key = make_key("geometry", file_hash)
    cached = cache_get(key)
    if cached is None:
        cached = expensive_operation()
        cache_set(key, cached, ttl=3600)
"""
import hashlib
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
_redis_client = None

if REDIS_URL:
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        _redis_client.ping()
        logger.info(f"OK: Redis cache connected ({REDIS_URL})")
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}) — caching disabled. Set REDIS_URL to enable.")
        _redis_client = None
else:
    logger.debug("REDIS_URL not set — caching disabled")


def make_key(*parts: str) -> str:
    """Build a namespaced cache key from arbitrary string parts."""
    return "cadfactory:" + ":".join(parts)


def make_hash_key(prefix: str, data: bytes) -> str:
    """Build a cache key using the SHA-256 hash of raw bytes (e.g. file content)."""
    digest = hashlib.sha256(data).hexdigest()[:32]
    return make_key(prefix, digest)


def cache_get(key: str) -> Optional[Any]:
    """
    Return the cached value for `key`, or None on miss / Redis unavailable.
    Stored values must be JSON-serialisable.
    """
    if _redis_client is None:
        return None
    try:
        raw = _redis_client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug(f"cache_get({key!r}) failed: {e}")
        return None


def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    """
    Store `value` under `key` with an expiry of `ttl` seconds.
    Returns True on success, False on failure / Redis unavailable.
    Value must be JSON-serialisable.
    """
    if _redis_client is None:
        return False
    try:
        _redis_client.set(key, json.dumps(value), ex=ttl)
        return True
    except Exception as e:
        logger.debug(f"cache_set({key!r}) failed: {e}")
        return False


def cache_delete(key: str) -> bool:
    """Delete a key. Returns True if the key existed, False otherwise."""
    if _redis_client is None:
        return False
    try:
        return bool(_redis_client.delete(key))
    except Exception as e:
        logger.debug(f"cache_delete({key!r}) failed: {e}")
        return False


def cache_available() -> bool:
    """Return True if the Redis client is connected and responding."""
    return _redis_client is not None
