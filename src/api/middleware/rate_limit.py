"""
Rate limiting middleware for API protection.
Simple in-memory rate limiter using token bucket algorithm.
"""

import time
from collections import defaultdict
from typing import Dict, Tuple


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)

    def check_rate_limit(self, client_id: str) -> Tuple[bool, int]:
        """
        Check if a client has exceeded the rate limit.
        Returns (is_allowed, remaining_requests).
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        # Check limit
        if len(self._requests[client_id]) >= self.max_requests:
            return False, 0

        # Allow request
        self._requests[client_id].append(now)
        remaining = self.max_requests - len(self._requests[client_id])
        return True, remaining


# Global rate limiter
rate_limiter = RateLimiter()
