"""Enterprise-grade structured logging system with comprehensive observability."""

import logging
import sys
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog
from structlog.typing import FilteringBoundLogger


# Context variables for tracing
request_id: ContextVar[str] = ContextVar("request_id", default="")
user_id: ContextVar[str] = ContextVar("user_id", default="")
operation: ContextVar[str] = ContextVar("operation", default="")


def add_request_context(logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add request context to log events."""
    if request_id.get():
        event_dict["request_id"] = request_id.get()
    if user_id.get():
        event_dict["user_id"] = user_id.get()
    if operation.get():
        event_dict["operation"] = operation.get()
    
    return event_dict


def add_timestamp(logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add ISO timestamp to log events."""
    event_dict["timestamp"] = time.time()
    event_dict["iso_timestamp"] = structlog.stdlib.get_logger().handlers[0].formatter.formatTime(
        logging.LogRecord("", 0, "", 0, "", (), None), "%Y-%m-%dT%H:%M:%S.%fZ"
    ) if hasattr(structlog.stdlib.get_logger(), 'handlers') and structlog.stdlib.get_logger().handlers else None
    return event_dict


def configure_logging(
    level: str = "INFO",
    enable_json: bool = True,
    enable_colors: bool = True,
    enable_dev_mode: bool = False,
) -> None:
    """Configure structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to use JSON formatting
        enable_colors: Whether to enable colored output
        enable_dev_mode: Whether to enable development-friendly formatting
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Configure processors based on environment
    processors = [
        structlog.contextvars.merge_contextvars,
        add_request_context,
        add_timestamp,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if enable_dev_mode:
        # Development mode: human-readable console output
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=enable_colors),
        ])
    else:
        # Production mode: structured JSON output
        if enable_json:
            processors.extend([
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ])
        else:
            processors.extend([
                structlog.dev.ConsoleRenderer(colors=enable_colors),
            ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add structured logging to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(
        self,
        logger: FilteringBoundLogger,
        operation: str,
        threshold_ms: float = 1000.0,
        **context: Any,
    ):
        self.logger = logger
        self.operation = operation
        self.threshold_ms = threshold_ms
        self.context = context
        self.start_time = 0.0
        
    def __enter__(self) -> "PerformanceLogger":
        self.start_time = time.time()
        self.logger.debug(
            "operation_started",
            operation=self.operation,
            **self.context,
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.time() - self.start_time) * 1000
        
        log_method = self.logger.warning if duration_ms > self.threshold_ms else self.logger.info
        
        log_method(
            "operation_completed",
            operation=self.operation,
            duration_ms=duration_ms,
            slow=duration_ms > self.threshold_ms,
            **self.context,
        )


def log_exception(
    logger: FilteringBoundLogger,
    exception: Exception,
    message: str = "Exception occurred",
    **context: Any,
) -> None:
    """Log an exception with full context.
    
    Args:
        logger: Structured logger instance
        exception: Exception to log
        message: Log message
        **context: Additional context
    """
    logger.exception(
        message,
        exception_type=type(exception).__name__,
        exception_message=str(exception),
        **context,
    )


def log_api_call(
    logger: FilteringBoundLogger,
    method: str,
    url: str,
    status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
    **context: Any,
) -> None:
    """Log an API call with standardized format.
    
    Args:
        logger: Structured logger instance
        method: HTTP method
        url: Request URL
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        **context: Additional context
    """
    logger.info(
        "api_call",
        http_method=method,
        url=url,
        status_code=status_code,
        duration_ms=duration_ms,
        **context,
    )


def log_business_event(
    logger: FilteringBoundLogger,
    event_type: str,
    event_data: Dict[str, Any],
    **context: Any,
) -> None:
    """Log a business event for analytics and monitoring.
    
    Args:
        logger: Structured logger instance
        event_type: Type of business event
        event_data: Event-specific data
        **context: Additional context
    """
    logger.info(
        "business_event",
        event_type=event_type,
        event_data=event_data,
        **context,
    )


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self):
        self.logger = get_logger("security")
    
    def log_authentication_attempt(
        self,
        username: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Log an authentication attempt."""
        self.logger.warning(
            "authentication_attempt",
            username=username,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            **context,
        )
    
    def log_authorization_failure(
        self,
        username: str,
        resource: str,
        action: str,
        **context: Any,
    ) -> None:
        """Log an authorization failure."""
        self.logger.warning(
            "authorization_failure",
            username=username,
            resource=resource,
            action=action,
            **context,
        )
    
    def log_suspicious_activity(
        self,
        activity_type: str,
        details: Dict[str, Any],
        **context: Any,
    ) -> None:
        """Log suspicious activity."""
        self.logger.error(
            "suspicious_activity",
            activity_type=activity_type,
            details=details,
            **context,
        )


# Global security logger instance
security_logger = SecurityLogger()


def setup_logging_from_config():
    """Setup logging based on application configuration."""
    from ..config import settings
    
    configure_logging(
        level=settings.monitoring.log_level.value,
        enable_json=settings.monitoring.structured_logging,
        enable_colors=not settings.is_production(),
        enable_dev_mode=settings.is_development(),
    )