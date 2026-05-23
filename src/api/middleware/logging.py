"""
Request logging middleware for FastAPI.
Logs all API requests with timing and response codes.
"""

import time
from fastapi import Request
from src.utils.logger import logger


async def log_requests_middleware(request: Request, call_next):
    """Log all incoming API requests with timing."""
    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration_ms:.1f}ms"
    )

    # Add duration header
    response.headers["X-Process-Time-MS"] = str(int(duration_ms))
    return response
