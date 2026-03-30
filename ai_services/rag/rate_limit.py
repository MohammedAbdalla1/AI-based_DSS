"""
Simple in-memory per-tenant rate limiting for the RAG query endpoint.
"""

from __future__ import annotations

from collections import defaultdict, deque
from time import time

from config import get_settings


class RateLimitError(Exception):
    pass


_REQUESTS: dict[str, deque[float]] = defaultdict(deque)


def enforce_tenant_rate_limit(tenant_id: str) -> None:
    settings = get_settings()
    limit = settings.rag_rate_limit_per_minute
    now = time()
    window_start = now - 60
    bucket = _REQUESTS[tenant_id]

    while bucket and bucket[0] < window_start:
        bucket.popleft()

    if len(bucket) >= limit:
        raise RateLimitError("Rate limit exceeded")

    bucket.append(now)


def reset_rate_limits() -> None:
    _REQUESTS.clear()
