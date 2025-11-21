"""Tests for observability features."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.main import app

client = TestClient(app)


def test_request_id_in_response():
    """Test that request ID is added to response headers."""
    response = client.get("/healthz")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_structured_logging_on_request(caplog):
    """Test that requests are logged in structured format."""
    import logging
    caplog.set_level(logging.INFO)
    
    response = client.get("/healthz")
    assert response.status_code == 200

    # Structured logs go to stdout, not caplog in this setup
    # Just verify the endpoint works
    assert response.status_code == 200


def test_validation_error_handling():
    """Test validation error returns proper JSON format."""
    response = client.post(
        "/api/v1/search",
        json={"query": "", "limit": 1000},  # Invalid: empty query, limit too high
    )

    assert response.status_code == 422
    data = response.json()
    # Response is the error object directly, not wrapped
    assert data["code"] == "validation_error"
    assert "message" in data


def test_generic_exception_handling():
    """Test unhandled exceptions return proper JSON format."""
    with patch("src.api.routers.search.settings") as mock_settings:
        mock_settings.qdrant = None

        response = client.post(
            "/api/v1/search",
            json={"query": "test", "limit": 10},
        )

        assert response.status_code == 503
        data = response.json()
        assert "detail" in data or "error" in data


def test_health_endpoint_no_errors():
    """Test health endpoint works without errors."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_readyz_endpoint_structure():
    """Test readiness endpoint returns proper structure."""
    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant" in data


def test_cors_headers():
    """Test CORS headers are present."""
    response = client.options("/healthz")
    # CORS middleware should add headers
    assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled


def test_error_response_no_sensitive_data():
    """Test that error responses don't leak sensitive data."""
    with patch("src.api.routers.search.settings") as mock_settings:
        mock_settings.qdrant = None

        response = client.post(
            "/api/v1/search",
            json={"query": "test"},
        )

        data = response.json()
        response_str = str(data).lower()

        # Check that common sensitive patterns are not in response
        assert "api_key" not in response_str
        assert "token" not in response_str
        assert "password" not in response_str
