"""Tests for ingestion pipeline."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.ingestion.pipeline import IngestionPipeline


@pytest.mark.asyncio
async def test_ingestion_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = IngestionPipeline()
    
    with patch.object(pipeline.qdrant_store, 'initialize', new_callable=AsyncMock):
        await pipeline.initialize()
        pipeline.qdrant_store.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_repo_success():
    """Test successful repository ingestion."""
    pipeline = IngestionPipeline()
    
    mock_repo_data = {
        "name": "test-repo",
        "full_name": "owner/test-repo",
        "description": "Test repository",
        "url": "https://github.com/owner/test-repo",
        "stars": 100,
        "language": "Python",
        "topics": ["test"],
        "created_at": "2024-01-01",
        "updated_at": "2024-01-02",
    }
    
    mock_enriched = {**mock_repo_data, "summary": "Test summary", "tags": ["test"]}
    
    with patch.object(pipeline, '_fetch_repo_metadata', new_callable=AsyncMock, return_value=mock_repo_data):
        with patch.object(pipeline, '_fetch_readme', new_callable=AsyncMock, return_value="# Test README"):
            with patch.object(pipeline.summarizer, 'summarize', new_callable=AsyncMock, return_value=mock_enriched):
                with patch.object(pipeline.qdrant_store, 'store_repositories', new_callable=AsyncMock):
                    result = await pipeline.ingest_repo("https://github.com/owner/test-repo")
                    
                    assert result["full_name"] == "owner/test-repo"
                    assert result["summary"] == "Test summary"
                    pipeline.qdrant_store.store_repositories.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_repo_invalid_url():
    """Test ingestion with invalid URL."""
    pipeline = IngestionPipeline()
    
    with pytest.raises(ValueError, match="Invalid repository URL"):
        await pipeline.ingest_repo("invalid-url")


@pytest.mark.asyncio
async def test_fetch_repo_metadata():
    """Test fetching repository metadata."""
    pipeline = IngestionPipeline()
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "test-repo",
        "full_name": "owner/test-repo",
        "description": "Test",
        "html_url": "https://github.com/owner/test-repo",
        "stargazers_count": 50,
        "language": "Python",
        "topics": ["test"],
        "created_at": "2024-01-01",
        "updated_at": "2024-01-02",
    }
    
    with patch("src.ingestion.pipeline.settings") as mock_settings:
        mock_settings.github = MagicMock()
        mock_settings.github.api_url = "https://api.github.com"
        mock_settings.github.token.get_secret_value.return_value = "test_token"
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await pipeline._fetch_repo_metadata("owner/test-repo")
            
            assert result["name"] == "test-repo"
            assert result["stars"] == 50


@pytest.mark.asyncio
async def test_reindex(tmp_path):
    """Test reindexing from JSON file."""
    pipeline = IngestionPipeline()
    
    # Create test repos file
    repos_file = tmp_path / "repos.json"
    test_repos = [
        {
            "name": "repo1",
            "full_name": "owner/repo1",
            "description": "Test 1",
            "readme": "# Repo 1",
        },
        {
            "name": "repo2",
            "full_name": "owner/repo2",
            "description": "Test 2",
            "readme": "# Repo 2",
            "summary": "Already has summary",
            "tags": ["test"],
        },
    ]
    repos_file.write_text(json.dumps(test_repos))
    
    mock_enriched = {**test_repos[0], "summary": "Generated summary", "tags": ["test"]}
    
    with patch.object(pipeline.summarizer, 'summarize', new_callable=AsyncMock, return_value=mock_enriched):
        with patch.object(pipeline.qdrant_store, 'store_repositories', new_callable=AsyncMock):
            results = await pipeline.reindex(repos_file)
            
            assert len(results) == 2
            # First repo should be summarized
            assert results[0]["summary"] == "Generated summary"
            # Second repo already has summary
            assert results[1]["summary"] == "Already has summary"
            
            pipeline.qdrant_store.store_repositories.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_close():
    """Test pipeline cleanup."""
    pipeline = IngestionPipeline()
    
    with patch.object(pipeline.qdrant_store, 'close', new_callable=AsyncMock):
        await pipeline.close()
        pipeline.qdrant_store.close.assert_called_once()
