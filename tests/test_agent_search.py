import pytest
from unittest.mock import patch, Mock
from src.agent.search import search_github, get_local_repos

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")

class MockAsyncClient:
    def __init__(self, response_data):
        self.response_data = response_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get(self, url, **kwargs):
        return MockResponse(self.response_data)

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

    with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
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

    with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
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
