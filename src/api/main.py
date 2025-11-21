"""FastAPI application for Oh My Repos search API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import health, search
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="Oh My Repos API",
    description="Semantic search API for GitHub repositories",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
if settings.security.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routers
app.include_router(health.router, tags=["health"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
