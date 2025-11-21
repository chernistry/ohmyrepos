"""Utility functions for API."""

import structlog

logger = structlog.get_logger()


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.
    
    Uses a simple heuristic: ~4 characters per token.
    For production, consider using tiktoken for accurate counting.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


def log_token_usage(operation: str, input_tokens: int, output_tokens: int = 0):
    """Log token usage for an operation.
    
    Args:
        operation: Name of the operation (e.g., "search", "summarize")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    """
    total_tokens = input_tokens + output_tokens
    
    logger.info(
        "token_usage",
        operation=operation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    
    return total_tokens
