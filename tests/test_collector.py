"""Tests for the RepoCollector class."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.core.collector import RepoCollector


@pytest.fixture
def mock_response():
    """Create a mock response for testing."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = [
        {
            "full_name": "owner/repo",
            "html_url": "https://github.com/owner/repo",
            "description": "Test repository",
            "topics": ["test", "python"],
            "language": "Python",
            "stargazers_count": 10,
            "forks_count": 5,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z",
        }
    ]
    return response


@pytest.fixture
def mock_readme_response():
    """Create a mock README response for testing."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "content": "IyBUZXN0IFJlcG9zaXRvcnkKCkEgdGVzdCByZXBvc2l0b3J5IGZvciB0ZXN0aW5nLg=="  # "# Test Repository\n\nA test repository for testing."
    }
    return response


@pytest.mark.asyncio
async def test_fetch_repos(mock_response):
    """Test fetching repositories."""
    with patch.object(RepoCollector, "_safe_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        
        collector = RepoCollector(username="test", token="test_token")
        repos = await collector._fetch_repos()
        
        assert len(repos) == 1
        assert repos[0]["name"] == "owner/repo"
        assert repos[0]["description"] == "Test repository"
        assert repos[0]["topics"] == ["test", "python"]
        
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_readme(mock_readme_response):
    """Test fetching README content."""
    with patch.object(RepoCollector, "_safe_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_readme_response
        
        collector = RepoCollector(username="test", token="test_token")
        readme = await collector._fetch_readme("owner/repo")
        
        assert "Test Repository" in readme
        assert "test repository for testing" in readme
        
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_collect_starred_repos(mock_response, mock_readme_response):
    """Test collecting starred repositories."""
    with patch.object(RepoCollector, "_fetch_repos", new_callable=AsyncMock) as mock_fetch_repos:
        with patch.object(RepoCollector, "_fetch_readme", new_callable=AsyncMock) as mock_fetch_readme:
            mock_fetch_repos.return_value = [
                {
                    "name": "owner/repo",
                    "html_url": "https://github.com/owner/repo",
                    "description": "Test repository",
                }
            ]
            mock_fetch_readme.return_value = "# Test Repository\n\nA test repository for testing."
            
            collector = RepoCollector(username="test", token="test_token")
            repos = await collector.collect_starred_repos()
            
            assert len(repos) == 1
            assert repos[0]["name"] == "owner/repo"
            assert repos[0]["readme"] == "# Test Repository\n\nA test repository for testing."
            
            mock_fetch_repos.assert_called_once()
            mock_fetch_readme.assert_called_once_with("owner/repo") 