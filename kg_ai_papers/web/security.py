# kg_ai_papers/web/security.py

from __future__ import annotations

import time
from typing import Dict, Tuple

from fastapi import Header, HTTPException, status, Request, Depends

from kg_ai_papers.config.settings import settings


# -------------------------------
# API key auth
# -------------------------------

def api_key_auth(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """
    Simple header-based API key auth.
    If settings.API_KEY is None, auth is disabled.
    """
    expected = settings.API_KEY.get_secret_value() if settings.API_KEY else None
    if expected is None:
        # auth disabled
        return

    if x_api_key is None or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# -------------------------------
# Very simple in-memory rate limiter
# -------------------------------

# key -> (window_start, count)
_RATE_LIMIT_STATE: Dict[str, Tuple[float, int]] = {}

RATE_LIMIT_MAX_REQUESTS = 60      # requests
RATE_LIMIT_WINDOW_SECONDS = 60.0  # 1 minute


def rate_limiter(request: Request):
    """
    Naive in-memory rate limiter by client host.
    Good enough for a dev box / single process.

    For something real, you'd use Redis or a gateway like Traefik/NGINX.
    """
    # identify client by IP
    client_host = request.client.host if request.client else "unknown"

    now = time.time()
    window_start, count = _RATE_LIMIT_STATE.get(client_host, (now, 0))

    # if outside window, reset
    if now - window_start >= RATE_LIMIT_WINDOW_SECONDS:
        window_start, count = now, 0

    count += 1

    if count > RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
        )

    _RATE_LIMIT_STATE[client_host] = (window_start, count)
