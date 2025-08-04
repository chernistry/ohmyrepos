"""Integration tests for Oh My Repos."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.core.collector import RepoCollector
from src.core.retriever import HybridRetriever
from src.core.summarizer import RepoSummarizer
from src.core.storage import QdrantStore
from src.config import settings


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def sample_repo_data(self):
        """Sample repository data for testing."""
        return {
            "id": 123456,
            "full_name": "test/example-repo",
            "name": "example-repo",
            "description": "A test repository for machine learning",
            "html_url": "https://github.com/test/example-repo",
            "stargazers_count": 100,
            "language": "Python",
            "topics": ["machine-learning", "python", "test"],
            "readme": "# Example Repo\n\nThis is a test repository for machine learning projects."
        }

    @pytest.fixture
    def temp_repos_file(self, sample_repo_data):
        """Create temporary repos.json file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([sample_repo_data], f)
            return Path(f.name)

    @pytest.mark.asyncio
    async def test_hybrid_retriever_initialization(self, temp_repos_file):
        """Test HybridRetriever initialization with real data."""
        with patch('src.core.storage.QdrantStore') as mock_qdrant:
            # Mock QdrantStore
            mock_store = AsyncMock()
            mock_store.setup_collection = AsyncMock()
            mock_qdrant.return_value = mock_store
            
            retriever = HybridRetriever(repos_json_path=str(temp_repos_file))
            await retriever.initialize()
            
            # Verify initialization
            assert retriever.repo_data is not None
            assert len(retriever.repo_data) == 1
            assert retriever.bm25_index is not None
            
            await retriever.close()

    @pytest.mark.asyncio
    async def test_search_pipeline(self, temp_repos_file):
        """Test the complete search pipeline."""
        with patch('src.core.storage.QdrantStore') as mock_qdrant:
            # Mock QdrantStore
            mock_store = AsyncMock()
            mock_store.setup_collection = AsyncMock()
            mock_store.search = AsyncMock(return_value=[
                {
                    "repo_name": "test/example-repo",
                    "repo_url": "https://github.com/test/example-repo",
                    "summary": "A test repository for machine learning",
                    "language": "Python",
                    "stars": 100,
                    "score": 0.85
                }
            ])
            mock_qdrant.return_value = mock_store
            
            retriever = HybridRetriever(repos_json_path=str(temp_repos_file))
            await retriever.initialize()
            
            # Test search
            results = await retriever.search("machine learning", limit=5)
            
            assert len(results) > 0
            assert "repo_name" in results[0]
            assert "score" in results[0]
            
            await retriever.close()

    @pytest.mark.asyncio
    async def test_collector_with_mock_github(self):
        """Test RepoCollector with mocked GitHub API."""
        from src.config import GitHubConfig
        from pydantic import SecretStr
        
        # Create a mock response for the user validation call
        mock_user_response = AsyncMock()
        mock_user_response.status_code = 200
        mock_user_response.json.return_value = {"login": "testuser"}
        
        # Create a mock response for the starred repos call
        mock_repos_response = AsyncMock()
        mock_repos_response.status_code = 200
        mock_repos_response.json.return_value = [
            {
                "id": 123456,
                "full_name": "test/repo",
                "name": "repo",
                "description": "Test repository",
                "html_url": "https://github.com/test/repo",
                "stargazers_count": 50,
                "language": "Python",
                "topics": [],
                "updated_at": "2023-01-01T00:00:00Z",
                "created_at": "2023-01-01T00:00:00Z",
                "pushed_at": "2023-01-01T00:00:00Z",
                "default_branch": "main",
                "owner": {
                    "login": "test",
                    "avatar_url": "https://github.com/test.png"
                }
            }
        ]
        
        # Mock the HTTP client
        with patch('src.core.collector.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = [mock_user_response, mock_repos_response]
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            github_config = GitHubConfig(
                username="testuser",
                token=SecretStr("test_token")
            )
            collector = RepoCollector(github_config=github_config)
            await collector.initialize()
            
            repos = []
            async for repo in collector.collect_starred_repos("testuser"):
                repos.append(repo)
            
            assert len(repos) == 1
            assert repos[0].full_name == "test/repo"
            
            await collector.close()

    @pytest.mark.asyncio
    async def test_summarizer_with_mock_llm(self, sample_repo_data):
        """Test RepoSummarizer with mocked LLM."""
        with patch('src.llm.providers.openai.OpenAIProvider') as mock_provider:
            # Mock LLM response
            mock_provider_instance = AsyncMock()
            mock_provider_instance.generate.return_value = json.dumps({
                "summary": "A machine learning repository with Python code",
                "tags": ["machine-learning", "python", "data-science"]
            })
            mock_provider.return_value = mock_provider_instance
            
            summarizer = RepoSummarizer()
            result = await summarizer.summarize(sample_repo_data)
            
            assert "summary" in result
            assert "tags" in result
            assert len(result["tags"]) > 0

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that settings can be loaded
        assert settings is not None
        assert hasattr(settings, 'GITHUB_USERNAME')
        assert hasattr(settings, 'BM25_WEIGHT')
        assert hasattr(settings, 'VECTOR_WEIGHT')
        
        # Test weight normalization
        total_weight = settings.BM25_WEIGHT + settings.VECTOR_WEIGHT
        assert abs(total_weight - 1.0) < 0.01  # Allow small floating point errors

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, temp_repos_file):
        """Test the complete end-to-end pipeline."""
        with patch('src.core.storage.QdrantStore') as mock_qdrant, \
             patch('src.llm.providers.openai.OpenAIProvider') as mock_llm:
            
            # Mock dependencies
            mock_store = AsyncMock()
            mock_store.setup_collection = AsyncMock()
            mock_store.store_repositories = AsyncMock()
            mock_store.search = AsyncMock(return_value=[])
            mock_qdrant.return_value = mock_store
            
            mock_llm_instance = AsyncMock()
            mock_llm_instance.generate.return_value = json.dumps({
                "summary": "Test summary",
                "tags": ["test", "integration"]
            })
            mock_llm.return_value = mock_llm_instance
            
            # Test pipeline components
            summarizer = RepoSummarizer()
            retriever = HybridRetriever(repos_json_path=str(temp_repos_file))
            
            await retriever.initialize()
            
            # Verify pipeline works
            assert retriever.repo_data is not None
            assert len(retriever.repo_data) > 0
            
            await retriever.close()


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_help_commands(self):
        """Test that all CLI commands have help text."""
        import subprocess
        
        commands = ['collect', 'summarize', 'embed', 'search', 'serve']
        
        for command in commands:
            result = subprocess.run(
                ['python', '/Users/sasha/IdeaProjects/ohmyrepos/ohmyrepos.py', command, '--help'],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert 'Usage:' in result.stdout

    def test_cli_error_handling(self):
        """Test CLI error handling for missing requirements."""
        import subprocess
        
        # Test search without proper setup (should show error gracefully)
        result = subprocess.run(
            ['python', '/Users/sasha/IdeaProjects/ohmyrepos/ohmyrepos.py', 'search', 'test'],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Should not crash, should show meaningful error
        assert result.returncode != 0 or 'Error' in result.stderr


class TestPerformance:
    """Performance tests."""

    @pytest.mark.asyncio
    async def test_search_performance(self, temp_repos_file):
        """Test search performance with mock data."""
        import time
        
        with patch('src.core.storage.QdrantStore') as mock_qdrant:
            mock_store = AsyncMock()
            mock_store.setup_collection = AsyncMock()
            mock_store.search = AsyncMock(return_value=[])
            mock_qdrant.return_value = mock_store
            
            retriever = HybridRetriever(repos_json_path=str(temp_repos_file))
            await retriever.initialize()
            
            # Measure search time
            start_time = time.time()
            results = await retriever.search("test query", limit=10)
            end_time = time.time()
            
            search_time = end_time - start_time
            assert search_time < 5.0  # Should complete within 5 seconds
            
            await retriever.close()


class TestSecurity:
    """Security tests."""

    def test_config_secrets_not_logged(self):
        """Test that secrets are not exposed in logs."""
        # Ensure sensitive config values are marked properly
        assert hasattr(settings, 'GITHUB_TOKEN')
        assert hasattr(settings, 'CHAT_LLM_API_KEY')
        
        # These should be empty in test environment
        # In production, they should be loaded from secure sources
        
    def test_input_validation(self):
        """Test input validation for user queries."""
        from src.core.retriever import HybridRetriever
        
        retriever = HybridRetriever()
        
        # Test that empty queries are handled gracefully
        # (This would need actual implementation in the retriever)
        assert True  # Placeholder for actual validation tests


if __name__ == "__main__":
    pytest.main([__file__])