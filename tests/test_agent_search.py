import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.agent.search import search_github, get_local_repos, RepoMetadata

@pytest.mark.asyncio
async def test_search_github_success():
    mock_response = {
        "items": [
            {
                "name": "repo1",
                "full_name": "owner/repo1",
                "html_url": "https://github.com/owner/repo1",
                "description": "A test repo",
                "stargazers_count": 150,
                "language": "Python",
                "updated_at": "2024-01-01T00:00:00Z",
                "topics": ["test"]
            },
            {
                "name": "repo2",
                "full_name": "owner/repo2",
                "html_url": "https://github.com/owner/repo2",
                "description": "Another test repo",
                "stargazers_count": 200,
                "language": "Rust",
                "updated_at": "2024-01-02T00:00:00Z",
                "topics": []
            }
        ]
    }

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None
        
        mock_instance.get.return_value.status_code = 200
        mock_instance.get.return_value.json.return_value = mock_response
        mock_instance.get.return_value.raise_for_status = Mock()

        results = await search_github("test query")
        
        assert len(results) == 2
        assert results[0].name == "repo1"
        assert results[0].stars == 150
        assert results[1].full_name == "owner/repo2"

@pytest.mark.asyncio
async def test_search_github_dedup():
    mock_response = {
        "items": [
            {
                "name": "repo1",
                "full_name": "owner/repo1",
                "html_url": "https://github.com/owner/repo1",
                "stargazers_count": 150,
                "updated_at": "2024-01-01T00:00:00Z"
            },
            {
                "name": "repo2",
                "full_name": "owner/repo2",
                "html_url": "https://github.com/owner/repo2",
                "stargazers_count": 200,
                "updated_at": "2024-01-02T00:00:00Z"
            }
        ]
    }

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None

        mock_instance.get.return_value.status_code = 200
        mock_instance.get.return_value.json.return_value = mock_response
        mock_instance.get.return_value.raise_for_status = Mock()

        excluded = {"owner/repo1"}
        results = await search_github("test query", excluded_repos=excluded)
        
        assert len(results) == 1
        assert results[0].full_name == "owner/repo2"

@pytest.mark.asyncio
async def test_get_local_repos(tmp_path):
    repos_file = tmp_path / "repos.json"
    repos_file.write_text('[{"name": "repo1", "full_name": "owner/repo1"}, {"name": "repo2"}]', encoding="utf-8")
    
    repos = await get_local_repos(str(repos_file))
    assert "owner/repo1" in repos
    assert "repo2" in repos
