"""Test configuration and fixtures for Oh My Repos."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from src.config import Settings, GitHubConfig, LLMConfig, QdrantConfig, EmbeddingConfig


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    return Settings(
        environment="testing",
        debug=True,
        github=GitHubConfig(
            username="test_user",
            token=SecretStr("test_token"),
        ),
        llm=LLMConfig(
            provider="openai",
            model="deepseek/deepseek-r1-0528:free",
            api_key=SecretStr("test_key"),
        ),
        qdrant=QdrantConfig(
            url="http://localhost:6333",
            collection_name="test_repos",
        ),
        embedding=EmbeddingConfig(
            api_key=SecretStr("test_embedding_key"),
        ),
    )


@pytest.fixture
def sample_repository() -> Dict[str, Any]:
    """Sample repository data for testing."""
    return {
        "id": 123456789,
        "full_name": "test-user/awesome-ml-project",
        "name": "awesome-ml-project",
        "description": "An awesome machine learning project with Python and TensorFlow",
        "html_url": "https://github.com/test-user/awesome-ml-project",
        "clone_url": "https://github.com/test-user/awesome-ml-project.git",
        "ssh_url": "git@github.com:test-user/awesome-ml-project.git",
        "stargazers_count": 1250,
        "forks_count": 45,
        "watchers_count": 1250,
        "size": 2048,
        "default_branch": "main",
        "open_issues_count": 12,
        "language": "Python",
        "topics": ["machine-learning", "python", "tensorflow", "data-science"],
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True,
        "has_pages": False,
        "archived": False,
        "disabled": False,
        "visibility": "public",
        "pushed_at": "2023-12-01T10:30:00Z",
        "created_at": "2023-01-15T08:00:00Z",
        "updated_at": "2023-12-01T10:30:00Z",
        "license": {
            "key": "mit",
            "name": "MIT License",
            "spdx_id": "MIT",
            "url": "https://api.github.com/licenses/mit",
        },
        "readme": """# Awesome ML Project

A comprehensive machine learning project using Python and TensorFlow.

## Features

- Data preprocessing pipeline
- Multiple ML models (CNN, LSTM, Transformer)
- Model evaluation and comparison
- REST API for predictions
- Docker containerization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from awesome_ml import MLPipeline

pipeline = MLPipeline()
results = pipeline.train(data_path="data/train.csv")
```

## Contributing

Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.
""",
    }


@pytest.fixture
def sample_repositories() -> List[Dict[str, Any]]:
    """Multiple sample repositories for testing."""
    return [
        {
            "id": 123456789,
            "full_name": "test-user/ml-project",
            "name": "ml-project",
            "description": "Machine learning project with Python",
            "html_url": "https://github.com/test-user/ml-project",
            "stargazers_count": 1250,
            "language": "Python",
            "topics": ["machine-learning", "python"],
            "readme": "# ML Project\n\nA machine learning project.",
        },
        {
            "id": 987654321,
            "full_name": "test-user/web-scraper",
            "name": "web-scraper",
            "description": "Web scraping tool built with JavaScript",
            "html_url": "https://github.com/test-user/web-scraper",
            "stargazers_count": 450,
            "language": "JavaScript",
            "topics": ["web-scraping", "javascript", "nodejs"],
            "readme": "# Web Scraper\n\nA powerful web scraping tool.",
        },
        {
            "id": 555444333,
            "full_name": "test-user/data-viz",
            "name": "data-viz",
            "description": "Data visualization dashboard with React",
            "html_url": "https://github.com/test-user/data-viz",
            "stargazers_count": 890,
            "language": "TypeScript",
            "topics": ["data-visualization", "react", "typescript", "dashboard"],
            "readme": "# Data Visualization Dashboard\n\nInteractive charts and graphs.",
        },
    ]


@pytest.fixture
def temp_repos_file(sample_repositories) -> Path:
    """Create temporary repository JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_repositories, f, indent=2)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_github_response():
    """Mock GitHub API response."""
    def _create_response(data: Any, status_code: int = 200):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = data
        response.text = json.dumps(data)
        response.headers = {"Content-Type": "application/json"}
        return response
    
    return _create_response


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    def _create_response(summary: str, tags: List[str]):
        return json.dumps({
            "summary": summary,
            "tags": tags,
            "confidence": 0.95,
        })
    
    return _create_response


@pytest.fixture
def mock_embedding_response():
    """Mock embedding response."""
    def _create_response(dimension: int = 1024, count: int = 1):
        import random
        return {
            "embeddings": [
                [random.random() for _ in range(dimension)]
                for _ in range(count)
            ],
            "model": "jina-embeddings-v3",
            "dimension": dimension,
        }
    
    return _create_response


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = AsyncMock()
    
    # Mock collection operations
    client.get_collections.return_value = MagicMock(collections=[])
    client.create_collection.return_value = None
    client.get_collection.return_value = MagicMock(
        config=MagicMock(
            params=MagicMock(
                vectors=MagicMock(size=1024, distance="Cosine")
            )
        )
    )
    
    # Mock point operations
    client.upsert.return_value = None
    client.search.return_value = []
    client.scroll.return_value = ([], None)
    client.count.return_value = MagicMock(count=0)
    
    return client


@pytest.fixture
async def mock_httpx_client():
    """Mock httpx async client."""
    client = AsyncMock()
    
    # Default response
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {}
    response.text = "{}"
    response.is_success = True
    
    client.get.return_value = response
    client.post.return_value = response
    client.put.return_value = response
    client.delete.return_value = response
    
    return client


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    test_env = {
        "GITHUB_USERNAME": "test_user",
        "GITHUB_TOKEN": "test_token",
        "CHAT_LLM_API_KEY": "test_llm_key",
        "EMBEDDING_MODEL_API_KEY": "test_embedding_key",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "",
        "ENVIRONMENT": "testing",
    }
    
    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def performance_threshold():
    """Performance thresholds for testing."""
    return {
        "search_time_ms": 1000,
        "embedding_time_ms": 5000,
        "summarization_time_ms": 10000,
        "collection_time_ms": 30000,
    }


# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security
pytest.mark.slow = pytest.mark.slow
pytest.mark.requires_api = pytest.mark.requires_api


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that run quickly and don't require external resources"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "security: Security-focused tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "requires_api: Tests that require external API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to fast tests
        if "unit" not in [mark.name for mark in item.iter_markers()]:
            if (
                "mock" in item.name.lower()
                or "test_config" in item.name.lower()
                or item.fspath.basename.startswith("test_unit")
            ):
                item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in item.fspath.basename or "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ["end_to_end", "full_pipeline", "large_dataset"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset any global state or singletons here
    yield
    # Cleanup after test