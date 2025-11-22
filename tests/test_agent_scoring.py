import pytest
from unittest.mock import patch, Mock, AsyncMock
from src.agent.scoring import score_candidates, ScoredRepo
from src.agent.search import RepoMetadata
from src.agent.profile import InterestCluster

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
    def __init__(self, response_data=None):
        self.response_data = response_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get(self, url, **kwargs):
        # Mock README response
        return MockResponse({"content": "IyBSZWFkbWUgQ29udGVudA=="}) # "# Readme Content" in base64

    async def post(self, url, **kwargs):
        # Mock LLM response
        return MockResponse({
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"score": 8.5, "reasoning": "Good repo", "summary": "A summary"}\n```'
                    }
                }
            ]
        })

@pytest.mark.asyncio
async def test_score_candidates():
    candidates = [
        RepoMetadata(
            name="repo1",
            full_name="owner/repo1",
            url="http://github.com/owner/repo1",
            stars=100,
            updated_at="2024-01-01",
            topics=["python"],
            description="A test repo",
            language="Python"
        )
    ]
    profile = InterestCluster(
        name="Python",
        keywords=["python", "ai"],
        languages=["Python"],
        score=1.0
    )

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        # Mock settings to ensure LLM is configured
        with patch("src.config.settings.llm") as mock_llm:
            mock_llm.model = "test-model"
            mock_llm.base_url = "http://test-url"
            mock_llm.api_key.get_secret_value.return_value = "test-key"
            
            results = await score_candidates(candidates, profile)
            
            assert len(results) == 1
            assert results[0].score == 8.5
            assert results[0].reasoning == "Good repo"
            assert results[0].readme_summary == "A summary"

@pytest.mark.asyncio
async def test_score_candidates_no_llm():
    candidates = [
        RepoMetadata(
            name="repo1",
            full_name="owner/repo1",
            url="http://github.com/owner/repo1",
            stars=100,
            updated_at="2024-01-01",
            topics=["python"],
            description="A test repo",
            language="Python"
        )
    ]
    profile = InterestCluster(
        name="Python",
        keywords=["python", "ai"],
        languages=["Python"],
        score=1.0
    )

    # Mock settings where LLM is None
    with patch("src.config.settings.llm", None):
        results = await score_candidates(candidates, profile)
        assert len(results) == 0
