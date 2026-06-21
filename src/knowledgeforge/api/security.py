"""Production hardening: API-key auth, rate limiting, structured request logging.

All hardening is config-gated and OFF by default so existing tests pass unchanged:
  KF_API_KEY     — if set, every request requires header ``X-API-Key`` == value
                   (else 401). Unset → open (dev mode, no auth).
  KF_RATE_LIMIT  — if set, max requests per minute per client IP (else 429 with
                   Retry-After). Unset → rate limiting disabled.

Request logging + ``X-Request-ID`` propagation is always on (cheap, stdlib only).
Env vars are read per-request so behaviour can be toggled without rebuilding the app.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

ACCESS_LOGGER = "knowledgeforge.access"

# Paths that never require an API key (docs/health/openapi).
AUTH_EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/docs/oauth2-redirect",
    }
)

# Paths exempt from rate limiting.
RATE_LIMIT_EXEMPT_PATHS = frozenset({"/health"})


class _JsonFormatter(logging.Formatter):
    """Emit each access record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "access", None)
        if payload is None:
            payload = {"message": record.getMessage()}
        return json.dumps(payload, separators=(",", ":"))


def setup_logging() -> None:
    """Configure the access logger once (idempotent). Called from the lifespan."""
    logger = logging.getLogger(ACCESS_LOGGER)
    if any(getattr(h, "_kf_access", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler._kf_access = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _client_host(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Assign/propagate X-Request-ID and log one JSON line per request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        logging.getLogger(ACCESS_LOGGER).info(
            "request",
            extra={
                "access": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                    "client": _client_host(request),
                }
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter, gated by KF_RATE_LIMIT (per min per IP)."""

    WINDOW_SECONDS = 60.0

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        raw = os.environ.get("KF_RATE_LIMIT")
        if not raw or request.url.path in RATE_LIMIT_EXEMPT_PATHS:
            return await call_next(request)
        try:
            limit = int(raw)
        except ValueError:
            return await call_next(request)
        if limit <= 0:
            return await call_next(request)

        now = time.monotonic()
        key = _client_host(request)
        window = self._hits[key]
        cutoff = now - self.WINDOW_SECONDS
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= limit:
            retry_after = max(1, int(self.WINDOW_SECONDS - (now - window[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """API-key auth gated by KF_API_KEY. Unset → open. Exempts docs/health paths."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        expected = os.environ.get("KF_API_KEY")
        if not expected or request.url.path in AUTH_EXEMPT_PATHS:
            return await call_next(request)
        provided = request.headers.get("X-API-Key")
        # Constant-time comparison to avoid leaking the key via response timing.
        if not provided or not hmac.compare_digest(provided, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)
