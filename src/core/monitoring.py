"""Production monitoring and observability for Oh My Repos.

This module provides structured logging, metrics collection, and distributed tracing
for production environments.
"""

import time
import logging
import asyncio
from typing import Any, Dict, Optional, Callable
from contextlib import asynccontextmanager
from functools import wraps
import uuid

import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Prometheus metrics
# Используем один набор метрик с уникальными именами
# Исправлено дублирование метрик api_requests/api_requests_total
API_METRICS = Counter(
    "api_metrics", "API metrics", ["provider", "method", "status"]
)

REQUEST_DURATION_SECONDS = Histogram(
    "request_duration_seconds",
    "Request duration in seconds",
    ["provider", "method"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

ACTIVE_CONNECTIONS = Gauge(
    "active_connections_total", "Number of active connections", ["provider"]
)

EMBEDDING_OPERATIONS = Counter(
    "embedding_operations",
    "Embedding operations",
    ["provider", "model", "status"],
)

VECTOR_DB_OPERATIONS = Counter(
    "vector_db_operations",
    "Vector database operations",
    ["operation", "status"],
)

RERANK_OPERATIONS = Counter(
    "rerank_operations", "Rerank operations", ["provider", "status"]
)


class CorrelationID:
    """Thread-local correlation ID for request tracing."""

    _current_id: Optional[str] = None

    @classmethod
    def generate(cls) -> str:
        """Generate a new correlation ID."""
        cls._current_id = str(uuid.uuid4())
        return cls._current_id

    @classmethod
    def get(cls) -> Optional[str]:
        """Get the current correlation ID."""
        return cls._current_id

    @classmethod
    def set(cls, correlation_id: str) -> None:
        """Set the correlation ID."""
        cls._current_id = correlation_id


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger with correlation ID."""
    logger = structlog.get_logger(name)
    correlation_id = CorrelationID.get()
    if correlation_id:
        logger = logger.bind(correlation_id=correlation_id)
    return logger


class MetricsCollector:
    """Collects and exposes application metrics."""

    def __init__(self, port: int = 8000):
        self.port = port
        self._server_started = False

    def start_server(self) -> None:
        """Start the Prometheus metrics server."""
        if not self._server_started:
            try:
                start_http_server(self.port)
                self._server_started = True
                logger = get_logger(__name__)
                logger.info(f"Metrics server started on port {self.port}")
            except OSError as e:
                logger = get_logger(__name__)
                logger.warning(f"Failed to start metrics server: {e}")

    @staticmethod
    def record_api_request(provider: str, method: str, status: str) -> None:
        """Record an API request."""
        API_METRICS.labels(provider=provider, method=method, status=status).inc()

    @staticmethod
    def record_request_duration(provider: str, method: str, duration: float) -> None:
        """Record request duration."""
        REQUEST_DURATION_SECONDS.labels(provider=provider, method=method).observe(
            duration
        )

    @staticmethod
    def set_active_connections(provider: str, count: int) -> None:
        """Set the number of active connections."""
        ACTIVE_CONNECTIONS.labels(provider=provider).set(count)

    @staticmethod
    def record_embedding_operation(provider: str, model: str, status: str) -> None:
        """Record an embedding operation."""
        EMBEDDING_OPERATIONS.labels(
            provider=provider, model=model, status=status
        ).inc()

    @staticmethod
    def record_vector_db_operation(operation: str, status: str) -> None:
        """Record a vector database operation."""
        VECTOR_DB_OPERATIONS.labels(operation=operation, status=status).inc()

    @staticmethod
    def record_rerank_operation(provider: str, status: str) -> None:
        """Record a rerank operation."""
        RERANK_OPERATIONS.labels(provider=provider, status=status).inc()


# Global metrics collector instance
metrics = MetricsCollector()


def monitored_operation(
    provider: str,
    method: str,
    record_embedding: bool = False,
    embedding_model: Optional[str] = None,
    record_vector_db: bool = False,
    vector_db_operation: Optional[str] = None,
    record_rerank: bool = False,
):
    """Decorator for monitoring operations with metrics and logging."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            correlation_id = CorrelationID.generate()
            logger = get_logger(func.__module__)

            start_time = time.time()
            status = "success"

            logger.info(
                f"Starting {provider}.{method}",
                operation=f"{provider}.{method}",
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                logger.error(
                    f"Operation {provider}.{method} failed",
                    operation=f"{provider}.{method}",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
            finally:
                duration = time.time() - start_time

                # Record metrics
                metrics.record_api_request(provider, method, status)
                metrics.record_request_duration(provider, method, duration)

                if record_embedding and embedding_model:
                    metrics.record_embedding_operation(
                        provider, embedding_model, status
                    )

                if record_vector_db and vector_db_operation:
                    metrics.record_vector_db_operation(vector_db_operation, status)

                if record_rerank:
                    metrics.record_rerank_operation(provider, status)

                logger.info(
                    f"Completed {provider}.{method}",
                    operation=f"{provider}.{method}",
                    duration=duration,
                    status=status,
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            correlation_id = CorrelationID.generate()
            logger = get_logger(func.__module__)

            start_time = time.time()
            status = "success"

            logger.info(
                f"Starting {provider}.{method}",
                operation=f"{provider}.{method}",
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                logger.error(
                    f"Operation {provider}.{method} failed",
                    operation=f"{provider}.{method}",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
            finally:
                duration = time.time() - start_time

                # Record metrics
                metrics.record_api_request(provider, method, status)
                metrics.record_request_duration(provider, method, duration)

                if record_embedding and embedding_model:
                    metrics.record_embedding_operation(
                        provider, embedding_model, status
                    )

                if record_vector_db and vector_db_operation:
                    metrics.record_vector_db_operation(vector_db_operation, status)

                if record_rerank:
                    metrics.record_rerank_operation(provider, status)

                logger.info(
                    f"Completed {provider}.{method}",
                    operation=f"{provider}.{method}",
                    duration=duration,
                    status=status,
                )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@asynccontextmanager
async def connection_monitor(provider: str, initial_count: int = 0):
    """Context manager for monitoring active connections."""
    logger = get_logger(__name__)

    try:
        metrics.set_active_connections(provider, initial_count + 1)
        logger.debug(f"Connection opened for {provider}")
        yield
    finally:
        metrics.set_active_connections(provider, max(0, initial_count))
        logger.debug(f"Connection closed for {provider}")


class HealthCheck:
    """Health check utilities for production deployments."""

    def __init__(self):
        self.checks: Dict[str, Callable] = {}

    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func

    async def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        logger = get_logger(__name__)

        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()

                results[name] = {"status": "healthy", "details": result}
            except Exception as e:
                logger.error(f"Health check {name} failed", error=str(e))
                results[name] = {"status": "unhealthy", "error": str(e)}

        overall_status = (
            "healthy"
            if all(check["status"] == "healthy" for check in results.values())
            else "unhealthy"
        )

        return {"status": overall_status, "checks": results, "timestamp": time.time()}


# Global health check instance
health_check = HealthCheck()


def setup_monitoring(metrics_port: int = 8000) -> None:
    """Setup monitoring infrastructure."""
    # Start metrics server
    metrics.port = metrics_port
    metrics.start_server()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger = get_logger(__name__)
    logger.info("Monitoring setup complete", metrics_port=metrics_port)
