"""
Rate Limiter Middleware
========================
IP-based rate limiting using slowapi.
"""

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
except ImportError:  # pragma: no cover
    class _NoOpLimiter:
        def limit(self, *args, **kwargs):
            return lambda f: f
    limiter = _NoOpLimiter()
