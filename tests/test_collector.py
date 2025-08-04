"""Comprehensive tests for the RepoCollector class."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from src.core.collector import RepoCollector, GitHubAPIError, GitHubRateLimiter
from src.core.models import RepositoryData
from src.config import GitHubConfig


@pytest.fixture
def github_config():
    """Create GitHub configuration for testing."""
    return GitHubConfig(
        username="test_user",
        token=SecretStr("test_token"),
        api_url="https://api.github.com",
    )


@pytest.fixture
def mock_github_repos():
    """Mock GitHub API repository response."""
    return [
        {
            "full_name": "test-user/awesome-project",
            "html_url": "https://github.com/test-user/awesome-project",
            "description": "An awesome project for testing",
            "topics": ["python", "machine-learning", "testing"],
            "language": "Python",
            "stargazers_count": 150,
            "forks_count": 25,
            "created_at": "2023-01-15T08:00:00Z",
            "updated_at": "2023-12-01T10:30:00Z",
        },
        {
            "full_name": "test-user/web-scraper",
            "html_url": "https://github.com/test-user/web-scraper",
            "description": "Web scraping utility",
            "topics": ["javascript", "web-scraping"],
            "language": "JavaScript",
            "stargazers_count": 75,
            "forks_count": 12,
            "created_at": "2023-06-10T12:00:00Z",
            "updated_at": "2023-11-20T14:15:00Z",
        },
    ]


@pytest.fixture
def mock_readme_response():
    """Mock README API response."""
    # Base64 encoded "# Awesome Project\n\nThis is a comprehensive machine learning project."
    content = "IyBBd2Vzb21lIFByb2plY3QKClRoaXMgaXMgYSBjb21wcmVoZW5zaXZlIG1hY2hpbmUgbGVhcm5pbmcgcHJvamVjdC4="
    return {
        "content": content,
        "encoding": "base64",
        "type": "file",
    }


@pytest.mark.unit
@pytest.mark.asyncio
class TestGitHubRateLimiter:
    """Test GitHub rate limiter functionality."""

    async def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = GitHubRateLimiter(requests_per_hour=3600)
        assert limiter.requests_per_hour == 3600
        assert limiter.min_interval == 1.0  # 3600 seconds / 3600 requests
        assert limiter.last_request_time == 0.0

    async def test_rate_limiter_acquire(self):
        """Test rate limiter acquire functionality."""
        limiter = GitHubRateLimiter(requests_per_hour=3600)  # 1 req/sec
        
        # First acquisition should be immediate
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        first_duration = asyncio.get_event_loop().time() - start_time
        assert first_duration < 0.1  # Should be nearly instantaneous
        
        # Second acquisition should wait
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        second_duration = asyncio.get_event_loop().time() - start_time
        assert second_duration >= 0.9  # Should wait ~1 second


@pytest.mark.unit
@pytest.mark.asyncio
class TestRepoCollectorInitialization:
    """Test RepoCollector initialization and setup."""

    async def test_collector_initialization(self, github_config):
        """Test collector initialization."""
        collector = RepoCollector(github_config=github_config)
        
        assert collector.config == github_config
        assert collector.max_concurrent == 10
        assert collector.chunk_size == 100
        assert collector.enable_readme_fetch is True
        assert not collector._initialized

    async def test_collector_without_config_fails(self):
        """Test that collector fails without configuration."""
        with patch('src.core.collector.settings') as mock_settings:
            mock_settings.github = None
            with pytest.raises(ValueError, match="GitHub configuration is required"):
                RepoCollector(github_config=None)

    async def test_collector_async_context_manager(self, github_config):
        """Test async context manager functionality."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = MagicMock(status_code=200)
            mock_client_class.return_value = mock_client
            
            async with RepoCollector(github_config=github_config) as collector:
                assert collector._initialized
                assert collector._client is not None
                assert collector._semaphore is not None
                assert collector._rate_limiter is not None


@pytest.mark.unit
@pytest.mark.asyncio 
class TestRepoCollectorConnection:
    """Test connection validation and error handling."""

    async def test_connection_validation_success(self, github_config):
        """Test successful connection validation."""
        collector = RepoCollector(github_config=github_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            await collector.initialize()
            
            assert collector._initialized
            mock_client.get.assert_called_with("/user")

    async def test_connection_validation_invalid_token(self, github_config):
        """Test connection validation with invalid token."""
        collector = RepoCollector(github_config=github_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_error = httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=MagicMock())
            mock_error.response.status_code = 401
            mock_client.get.side_effect = mock_error
            mock_client_class.return_value = mock_client
            
            with pytest.raises(GitHubAPIError, match="Invalid GitHub token"):
                await collector.initialize()

    async def test_connection_validation_rate_limit(self, github_config):
        """Test connection validation with rate limit."""
        collector = RepoCollector(github_config=github_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=MagicMock())
            mock_error.response.status_code = 403
            mock_client.get.side_effect = mock_error
            mock_client_class.return_value = mock_client
            
            with pytest.raises(GitHubAPIError, match="GitHub API rate limit exceeded"):
                await collector.initialize()


@pytest.mark.unit
class TestRepoCollectorDataProcessing:
    """Test repository data processing and conversion."""

    def test_convert_to_repo_data(self, github_config, mock_github_repos):
        """Test conversion of GitHub API data to RepositoryData."""
        collector = RepoCollector(github_config=github_config)
        repo_dict = mock_github_repos[0]
        
        repo_data = collector._convert_to_repo_data(repo_dict)
        
        assert isinstance(repo_data, RepositoryData)
        assert repo_data.repo_name == "test-user/awesome-project"
        assert repo_data.repo_url == "https://github.com/test-user/awesome-project"
        assert repo_data.description == "An awesome project for testing"
        assert repo_data.tags == ["python", "machine-learning", "testing"]
        assert repo_data.language == "Python"
        assert repo_data.stars == 150
        assert repo_data.forks == 25

    @pytest.mark.asyncio
    async def test_fetch_readme_success(self, github_config, mock_readme_response):
        """Test successful README fetching."""
        collector = RepoCollector(github_config=github_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            
            # Mock initialization
            mock_client.get.return_value = MagicMock(status_code=200)
            
            # Mock README request
            readme_response = MagicMock()
            readme_response.status_code = 200
            readme_response.json.return_value = mock_readme_response
            
            mock_client.get.side_effect = [
                MagicMock(status_code=200),  # User validation
                readme_response,  # README request
            ]
            mock_client_class.return_value = mock_client
            
            await collector.initialize()
            
            # Reset semaphore and rate limiter for clean test
            collector._semaphore = asyncio.Semaphore(1)
            collector._rate_limiter = GitHubRateLimiter()
            
            readme_content = await collector._fetch_readme("test-user/awesome-project")
            
            assert "Awesome Project" in readme_content
            assert "comprehensive machine learning project" in readme_content

    @pytest.mark.asyncio
    async def test_fetch_readme_not_found(self, github_config):
        """Test README fetch when README doesn't exist."""
        collector = RepoCollector(github_config=github_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            
            # Mock initialization
            mock_client.get.return_value = MagicMock(status_code=200)
            
            # Mock README request - 404
            readme_response = MagicMock()
            readme_response.status_code = 404
            
            mock_client.get.side_effect = [
                MagicMock(status_code=200),  # User validation
                readme_response,  # README request
            ]
            mock_client_class.return_value = mock_client
            
            await collector.initialize()
            
            # Reset semaphore and rate limiter for clean test
            collector._semaphore = asyncio.Semaphore(1)
            collector._rate_limiter = GitHubRateLimiter()
            
            readme_content = await collector._fetch_readme("test-user/no-readme")
            
            assert readme_content is None


@pytest.mark.integration
@pytest.mark.asyncio
class TestRepoCollectorIntegration:
    """Integration tests for RepoCollector."""

    async def test_collect_starred_repos_success(self, github_config, mock_github_repos, mock_readme_response):
        """Test complete starred repositories collection."""
        collector = RepoCollector(github_config=github_config, chunk_size=10)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            
            # Mock initialization
            mock_client.get.return_value = MagicMock(status_code=200)
            
            # Mock starred repos API calls
            starred_response = MagicMock()
            starred_response.status_code = 200
            starred_response.json.return_value = mock_github_repos
            starred_response.raise_for_status.return_value = None
            
            # Mock README API calls
            readme_response = MagicMock()
            readme_response.status_code = 200
            readme_response.json.return_value = mock_readme_response
            
            # Setup call sequence
            mock_client.get.side_effect = [
                MagicMock(status_code=200),  # User validation
                starred_response,  # First page of starred repos
                readme_response,  # README 1
                readme_response,  # README 2
            ]
            mock_client_class.return_value = mock_client
            
            await collector.initialize()
            
            # Collect repositories
            collected_repos = []
            async for repo in collector.collect_starred_repos():
                collected_repos.append(repo)
            
            assert len(collected_repos) == 2
            
            # Check first repo
            repo1 = collected_repos[0]
            assert isinstance(repo1, RepositoryData)
            assert repo1.repo_name == "test-user/awesome-project"
            assert repo1.stars == 150
            assert "comprehensive machine learning project" in repo1.summary
            
            # Check second repo
            repo2 = collected_repos[1]
            assert isinstance(repo2, RepositoryData)
            assert repo2.repo_name == "test-user/web-scraper"
            assert repo2.stars == 75

    async def test_collection_with_progress_callback(self, github_config, mock_github_repos):
        """Test collection with progress callback."""
        collector = RepoCollector(github_config=github_config, enable_readme_fetch=False)
        progress_calls = []
        
        async def progress_callback(count):
            progress_calls.append(count)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            
            # Mock initialization and starred repos
            starred_response = MagicMock()
            starred_response.status_code = 200
            starred_response.json.return_value = mock_github_repos
            starred_response.raise_for_status.return_value = None
            
            mock_client.get.side_effect = [
                MagicMock(status_code=200),  # User validation
                starred_response,  # Starred repos
            ]
            mock_client_class.return_value = mock_client
            
            await collector.initialize()
            
            # Collect with progress callback
            collected_repos = []
            async for repo in collector.collect_starred_repos(progress_callback=progress_callback):
                collected_repos.append(repo)
            
            assert len(collected_repos) == 2
            assert progress_calls == [1, 2]  # Should track progress

    async def test_collection_statistics_tracking(self, github_config, mock_github_repos):
        """Test that collection statistics are properly tracked."""
        collector = RepoCollector(github_config=github_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            
            starred_response = MagicMock()
            starred_response.status_code = 200
            starred_response.json.return_value = mock_github_repos
            starred_response.raise_for_status.return_value = None
            
            # Mock README responses
            readme_response = MagicMock()
            readme_response.status_code = 200
            readme_response.json.return_value = {"content": "VGVzdA==", "encoding": "base64"}
            
            mock_client.get.side_effect = [
                MagicMock(status_code=200),  # User validation
                starred_response,  # Starred repos
                readme_response,  # README 1
                readme_response,  # README 2
            ]
            mock_client_class.return_value = mock_client
            
            await collector.initialize()
            
            # Collect repositories
            collected_repos = []
            async for repo in collector.collect_starred_repos():
                collected_repos.append(repo)
            
            # Check statistics
            stats = await collector.get_collection_stats()
            assert stats["repos_fetched"] == 2
            assert stats["readmes_fetched"] == 2
            assert stats["errors"] == 0
            assert stats["start_time"] > 0
            assert stats["end_time"] > stats["start_time"]

    async def test_cleanup_on_close(self, github_config):
        """Test proper cleanup when collector is closed."""
        collector = RepoCollector(github_config=github_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = MagicMock(status_code=200)
            mock_client_class.return_value = mock_client
            
            await collector.initialize()
            assert collector._initialized
            
            await collector.close()
            assert not collector._initialized
            mock_client.aclose.assert_called_once()


@pytest.mark.performance
@pytest.mark.asyncio
class TestRepoCollectorPerformance:
    """Performance tests for RepoCollector."""

    async def test_concurrent_processing_limits(self, github_config):
        """Test that concurrent processing respects limits."""
        collector = RepoCollector(github_config=github_config, max_concurrent=2)
        
        assert collector.max_concurrent == 2
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = MagicMock(status_code=200)
            mock_client_class.return_value = mock_client
            
            await collector.initialize()
            
            # Semaphore should respect max_concurrent setting
            assert collector._semaphore._value == collector.config.max_concurrent_requests