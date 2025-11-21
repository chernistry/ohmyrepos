"""Rate limiting for API endpoints."""

import time
from collections import defaultdict
from typing import Dict, Tuple

import structlog
from fastapi import Request, HTTPException, status

logger = structlog.get_logger()


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self):
        # Store: {client_ip: {endpoint: [(timestamp, count)]}}
        self._requests: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._cleanup_interval = 300  # Clean up old entries every 5 minutes
        self._last_cleanup = time.time()

    def _cleanup(self):
        """Remove old request records."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        cutoff_time = current_time - 3600  # Keep last hour
        for client_data in self._requests.values():
            for endpoint, requests in client_data.items():
                client_data[endpoint] = [
                    (ts, count) for ts, count in requests if ts > cutoff_time
                ]

        self._last_cleanup = current_time

    def check_rate_limit(
        self, client_ip: str, endpoint: str, limit: int, window: int
    ) -> Tuple[bool, int]:
        """Check if request is within rate limit.
        
        Args:
            client_ip: Client IP address
            endpoint: API endpoint path
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        self._cleanup()

        current_time = time.time()
        cutoff_time = current_time - window

        # Get requests for this client and endpoint
        requests = self._requests[client_ip][endpoint]

        # Count requests within window
        recent_requests = [ts for ts, _ in requests if ts > cutoff_time]
        count = len(recent_requests)

        if count >= limit:
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                endpoint=endpoint,
                count=count,
                limit=limit,
                window=window,
            )
            return False, 0

        # Add current request
        requests.append((current_time, 1))
        remaining = limit - count - 1

        return True, remaining


# Global rate limiter instance
_rate_limiter: RateLimiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    return _rate_limiter


async def rate_limit_dependency(
    request: Request,
    limit: int = 60,
    window: int = 60,
):
    """FastAPI dependency for rate limiting.
    
    Args:
        request: FastAPI request
        limit: Maximum requests allowed
        window: Time window in seconds
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    limiter = get_rate_limiter()
    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path

    allowed, remaining = limiter.check_rate_limit(client_ip, endpoint, limit, window)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Maximum {limit} requests per {window} seconds.",
            },
        )

    # Add rate limit headers
    request.state.rate_limit_remaining = remaining
