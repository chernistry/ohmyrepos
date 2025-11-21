"""Budget tracking for token usage."""

import time
from typing import Optional

import structlog

logger = structlog.get_logger()


class TokenBudgetTracker:
    """Track token usage against daily budget."""

    def __init__(self, max_daily_tokens: Optional[int] = None):
        """Initialize tracker.
        
        Args:
            max_daily_tokens: Maximum tokens allowed per day (None = unlimited)
        """
        self.max_daily_tokens = max_daily_tokens
        self.daily_usage = 0
        self.last_reset = time.time()
        self._reset_interval = 86400  # 24 hours in seconds

    def _check_reset(self):
        """Reset counter if a day has passed."""
        current_time = time.time()
        if current_time - self.last_reset >= self._reset_interval:
            logger.info(
                "budget_reset",
                previous_usage=self.daily_usage,
                max_daily_tokens=self.max_daily_tokens,
            )
            self.daily_usage = 0
            self.last_reset = current_time

    def check_budget(self, tokens: int) -> bool:
        """Check if adding tokens would exceed budget.
        
        Args:
            tokens: Number of tokens to add
            
        Returns:
            True if within budget, False if would exceed
        """
        if self.max_daily_tokens is None:
            return True

        self._check_reset()
        return (self.daily_usage + tokens) <= self.max_daily_tokens

    def add_usage(self, tokens: int):
        """Add token usage to counter.
        
        Args:
            tokens: Number of tokens used
        """
        self._check_reset()
        self.daily_usage += tokens
        
        if self.max_daily_tokens:
            percentage = (self.daily_usage / self.max_daily_tokens) * 100
            logger.info(
                "budget_usage",
                daily_usage=self.daily_usage,
                max_daily_tokens=self.max_daily_tokens,
                percentage_used=round(percentage, 2),
            )

    def get_remaining(self) -> Optional[int]:
        """Get remaining tokens in budget.
        
        Returns:
            Remaining tokens, or None if unlimited
        """
        if self.max_daily_tokens is None:
            return None

        self._check_reset()
        return max(0, self.max_daily_tokens - self.daily_usage)


# Global budget tracker instance
_budget_tracker: Optional[TokenBudgetTracker] = None


def get_budget_tracker() -> TokenBudgetTracker:
    """Get or create global budget tracker."""
    global _budget_tracker
    
    if _budget_tracker is None:
        from src.config import settings
        
        # Get max daily tokens from env or default to None (unlimited)
        max_daily = None
        if hasattr(settings, 'max_daily_tokens'):
            max_daily = settings.max_daily_tokens
        
        _budget_tracker = TokenBudgetTracker(max_daily_tokens=max_daily)
        logger.info("budget_tracker_initialized", max_daily_tokens=max_daily)
    
    return _budget_tracker
