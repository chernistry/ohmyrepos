"""Tests for cost guardrails."""

import pytest
import time
from unittest.mock import patch

from src.api.budget import TokenBudgetTracker
from src.api.rate_limit import RateLimiter
from src.api.utils import estimate_tokens, log_token_usage


def test_estimate_tokens():
    """Test token estimation."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("test") == 1  # 4 chars = 1 token
    assert estimate_tokens("a" * 100) == 25  # 100 chars = 25 tokens


def test_log_token_usage(caplog):
    """Test token usage logging."""
    tokens = log_token_usage("test_op", 100, 50)
    assert tokens == 150


def test_budget_tracker_unlimited():
    """Test budget tracker with no limit."""
    tracker = TokenBudgetTracker(max_daily_tokens=None)
    
    assert tracker.check_budget(1000000)
    tracker.add_usage(1000000)
    assert tracker.get_remaining() is None


def test_budget_tracker_with_limit():
    """Test budget tracker with limit."""
    tracker = TokenBudgetTracker(max_daily_tokens=1000)
    
    # Within budget
    assert tracker.check_budget(500)
    tracker.add_usage(500)
    assert tracker.get_remaining() == 500
    
    # Still within budget
    assert tracker.check_budget(400)
    tracker.add_usage(400)
    assert tracker.get_remaining() == 100
    
    # Would exceed budget
    assert not tracker.check_budget(200)
    assert tracker.get_remaining() == 100


def test_budget_tracker_reset():
    """Test budget tracker resets after 24 hours."""
    tracker = TokenBudgetTracker(max_daily_tokens=1000)
    tracker.add_usage(900)
    
    # Simulate 24 hours passing
    tracker.last_reset = time.time() - 86401
    
    # Should reset
    tracker._check_reset()
    assert tracker.daily_usage == 0
    assert tracker.check_budget(1000)


def test_rate_limiter_allows_within_limit():
    """Test rate limiter allows requests within limit."""
    limiter = RateLimiter()
    
    # Should allow first 5 requests
    for i in range(5):
        allowed, remaining = limiter.check_rate_limit("127.0.0.1", "/test", 5, 60)
        assert allowed
        assert remaining == 4 - i


def test_rate_limiter_blocks_over_limit():
    """Test rate limiter blocks requests over limit."""
    limiter = RateLimiter()
    
    # Fill up the limit
    for _ in range(5):
        limiter.check_rate_limit("127.0.0.1", "/test", 5, 60)
    
    # Next request should be blocked
    allowed, remaining = limiter.check_rate_limit("127.0.0.1", "/test", 5, 60)
    assert not allowed
    assert remaining == 0


def test_rate_limiter_different_clients():
    """Test rate limiter tracks clients separately."""
    limiter = RateLimiter()
    
    # Client 1 uses up limit
    for _ in range(5):
        limiter.check_rate_limit("127.0.0.1", "/test", 5, 60)
    
    # Client 2 should still be allowed
    allowed, _ = limiter.check_rate_limit("192.168.1.1", "/test", 5, 60)
    assert allowed


def test_rate_limiter_different_endpoints():
    """Test rate limiter tracks endpoints separately."""
    limiter = RateLimiter()
    
    # Use up limit on /test
    for _ in range(5):
        limiter.check_rate_limit("127.0.0.1", "/test", 5, 60)
    
    # /other should still be allowed
    allowed, _ = limiter.check_rate_limit("127.0.0.1", "/other", 5, 60)
    assert allowed


def test_rate_limiter_window_expiry():
    """Test rate limiter respects time window."""
    limiter = RateLimiter()
    
    # Fill up limit
    for _ in range(5):
        limiter.check_rate_limit("127.0.0.1", "/test", 5, 1)  # 1 second window
    
    # Should be blocked
    allowed, _ = limiter.check_rate_limit("127.0.0.1", "/test", 5, 1)
    assert not allowed
    
    # Wait for window to expire
    time.sleep(1.1)
    
    # Should be allowed again
    allowed, _ = limiter.check_rate_limit("127.0.0.1", "/test", 5, 1)
    assert allowed
