"""FastAPI application for Oh My Repos search API."""

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from src.api.exceptions import generic_exception_handler, validation_exception_handler
from src.api.middleware import LoggingMiddleware
from src.api.routers import health, search
from src.config import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("application_startup", environment=settings.environment.value)
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title="Oh My Repos API",
    description="Semantic search API for GitHub repositories",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(LoggingMiddleware)

if settings.security.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Routers
app.include_router(health.router, tags=["health"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
