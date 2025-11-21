"""Health check endpoints."""

from fastapi import APIRouter, status
from pydantic import BaseModel

from src.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    environment: str


class ReadyResponse(BaseModel):
    """Readiness check response."""

    status: str
    qdrant: str


@router.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def healthz():
    """Lightweight health check."""
    return HealthResponse(
        status="ok",
        environment=settings.environment.value,
    )


@router.get("/readyz", response_model=ReadyResponse, status_code=status.HTTP_200_OK)
async def readyz():
    """Readiness check with dependency validation."""
    qdrant_status = "ok" if settings.qdrant else "not_configured"

    if settings.qdrant:
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(
                url=str(settings.qdrant.url),
                api_key=settings.qdrant.api_key.get_secret_value()
                if settings.qdrant.api_key
                else None,
                timeout=5,
            )
            client.get_collections()
            qdrant_status = "connected"
        except Exception as e:
            qdrant_status = f"error: {str(e)[:50]}"

    return ReadyResponse(status="ok", qdrant=qdrant_status)
