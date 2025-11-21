"""Tests for the verify_connections script."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.verify_connections import (
    check_github,
    check_jina_embeddings,
    check_ollama,
    check_openrouter,
    check_qdrant,
)


@pytest.mark.asyncio
async def test_check_github_success():
    """Test successful GitHub connection check."""
    with patch("scripts.verify_connections.settings") as mock_settings:
        mock_settings.github = MagicMock()
        mock_settings.github.api_url = "https://api.github.com"
        mock_settings.github.token.get_secret_value.return_value = "test_token"

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"login": "testuser"}

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            success, message = await check_github()
            assert success is True
            assert "testuser" in message


@pytest.mark.asyncio
async def test_check_github_no_config():
    """Test GitHub check with missing config."""
    with patch("scripts.verify_connections.settings") as mock_settings:
        mock_settings.github = None

        success, message = await check_github()
        assert success is False
        assert "not found" in message


@pytest.mark.asyncio
async def test_check_qdrant_success():
    """Test successful Qdrant connection check."""
    with patch("scripts.verify_connections.settings") as mock_settings:
        mock_settings.qdrant = MagicMock()
        mock_settings.qdrant.url = "http://localhost:6333"
        mock_settings.qdrant.api_key = None

        with patch("qdrant_client.QdrantClient") as mock_client:
            mock_collections = MagicMock()
            mock_collections.collections = [MagicMock(), MagicMock()]
            mock_client.return_value.get_collections.return_value = mock_collections

            success, message = await check_qdrant()
            assert success is True
            assert "2 collections" in message


@pytest.mark.asyncio
async def test_check_openrouter_success():
    """Test successful OpenRouter connection check."""
    with patch("scripts.verify_connections.settings") as mock_settings:
        mock_settings.llm = MagicMock()
        mock_settings.llm.base_url = "https://openrouter.ai/api/v1"
        mock_settings.llm.model = "test-model"
        mock_settings.llm.api_key.get_secret_value.return_value = "test_key"

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            success, message = await check_openrouter()
            assert success is True
            assert "test-model" in message


@pytest.mark.asyncio
async def test_check_jina_embeddings_success():
    """Test successful Jina embeddings check."""
    with patch("scripts.verify_connections.settings") as mock_settings:
        mock_settings.embedding = MagicMock()
        mock_settings.embedding.base_url = "https://api.jina.ai/v1/embeddings"
        mock_settings.embedding.model = "jina-embeddings-v3"
        mock_settings.embedding.api_key.get_secret_value.return_value = "test_key"

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            success, message = await check_jina_embeddings()
            assert success is True
            assert "jina-embeddings-v3" in message


@pytest.mark.asyncio
async def test_check_ollama_not_running():
    """Test Ollama check when service is not running."""
    with patch("scripts.verify_connections.settings") as mock_settings:
        mock_settings.ollama = MagicMock()
        mock_settings.ollama.base_url = "http://localhost:11434"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection refused")
            )

            success, message = await check_ollama()
            assert success is False
            assert "Not running" in message or "unreachable" in message
