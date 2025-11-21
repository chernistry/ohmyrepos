"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.main import app

client = TestClient(app)


def test_healthz():
    """Test health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data


def test_readyz_no_qdrant():
    """Test readiness check when Qdrant is not configured."""
    with patch("src.api.routers.health.settings") as mock_settings:
        mock_settings.qdrant = None

        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["qdrant"] == "not_configured"


def test_readyz_with_qdrant():
    """Test readiness check with Qdrant configured."""
    with patch("src.api.routers.health.settings") as mock_settings:
        mock_settings.qdrant = MagicMock()
        mock_settings.qdrant.url = "http://localhost:6333"
        mock_settings.qdrant.api_key = None

        with patch("qdrant_client.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value = MagicMock()

            response = client.get("/readyz")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["qdrant"] == "connected"


@pytest.mark.asyncio
async def test_search_endpoint_not_configured():
    """Test search endpoint when Qdrant is not configured."""
    with patch("src.api.routers.search.settings") as mock_settings:
        mock_settings.qdrant = None

        response = client.post(
            "/api/v1/search",
            json={"query": "test query", "limit": 10},
        )
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_endpoint_success():
    """Test successful search."""
    with patch("src.api.routers.search.settings") as mock_settings:
        mock_settings.qdrant = MagicMock()
        mock_settings.search = MagicMock()
        mock_settings.search.bm25_weight = 0.5
        mock_settings.search.vector_weight = 0.5
        mock_settings.search.bm25_variant = "plus"

        with patch("src.api.routers.search.QdrantStore") as mock_store_class:
            with patch("src.api.routers.search.HybridRetriever") as mock_retriever_class:
                # Mock retriever
                mock_retriever = AsyncMock()
                mock_retriever.initialize = AsyncMock()
                mock_retriever.search = AsyncMock(
                    return_value=[
                        {
                            "repo_name": "test/repo",
                            "full_name": "test/repo",
                            "description": "Test repository",
                            "summary": "A test repo",
                            "tags": ["test", "python"],
                            "language": "Python",
                            "stars": 100,
                            "url": "https://github.com/test/repo",
                            "score": 0.95,
                        }
                    ]
                )
                mock_retriever_class.return_value = mock_retriever

                response = client.post(
                    "/api/v1/search",
                    json={"query": "test query", "limit": 10},
                )
                assert response.status_code == 200
                data = response.json()
                assert data["query"] == "test query"
                assert len(data["results"]) == 1
                assert data["results"][0]["repo_name"] == "test/repo"
                assert data["total"] == 1


def test_search_endpoint_validation():
    """Test search endpoint input validation."""
    # Empty query
    response = client.post(
        "/api/v1/search",
        json={"query": "", "limit": 10},
    )
    assert response.status_code == 422

    # Invalid limit
    response = client.post(
        "/api/v1/search",
        json={"query": "test", "limit": 1000},
    )
    assert response.status_code == 422

    # Negative offset
    response = client.post(
        "/api/v1/search",
        json={"query": "test", "offset": -1},
    )
    assert response.status_code == 422
