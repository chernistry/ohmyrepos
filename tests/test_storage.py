"""Comprehensive tests for vector storage system."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.storage import (
    QdrantStore,
    StorageError,
    VectorDimensionError,
    ConnectionError,
)
from src.core.models import RepositoryData, SearchResult
from src.config import QdrantConfig


@pytest.mark.unit
class TestQdrantStoreInitialization:
    """Test QdrantStore initialization and configuration."""

    def test_init_with_config(self):
        """Test initialization with custom configuration."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="test_repos",
            vector_size=1024,
        )
        
        store = QdrantStore(qdrant_config=config)
        assert store.config == config
        assert store.collection_name == "test_repos"
        assert not store._initialized
        assert not store._collection_ready

    def test_init_without_config_fails(self):
        """Test that initialization fails without configuration."""
        with patch('src.core.storage.settings') as mock_settings:
            mock_settings.qdrant = None
            with pytest.raises(ValueError, match="Qdrant configuration is required"):
                QdrantStore(qdrant_config=None)

    def test_init_with_custom_batch_settings(self):
        """Test initialization with custom batch settings."""
        config = QdrantConfig(url="http://localhost:6333")
        store = QdrantStore(
            qdrant_config=config,
            batch_size=50,
            max_concurrent_batches=5
        )
        
        assert store.batch_size == 50
        assert store.max_concurrent_batches == 5
        assert store._semaphore._value == 5


@pytest.mark.unit
class TestQdrantStoreAsyncOperations:
    """Test async operations and context management."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_qdrant_client):
        """Test async context manager functionality."""
        config = QdrantConfig(url="http://localhost:6333")
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            async with QdrantStore(qdrant_config=config) as store:
                assert store._initialized
                assert store._client is not None

    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_qdrant_client):
        """Test successful initialization."""
        config = QdrantConfig(url="http://localhost:6333")
        store = QdrantStore(qdrant_config=config)
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            assert store._initialized
            assert store._collection_ready
            mock_qdrant_client.get_collections.assert_called()

    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test initialization failure handling."""
        config = QdrantConfig(url="http://localhost:6333")
        store = QdrantStore(qdrant_config=config)
        
        with patch('src.core.storage.AsyncQdrantClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_collections.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client
            
            with pytest.raises(ConnectionError, match="Failed to initialize Qdrant store"):
                await store.initialize()

    @pytest.mark.asyncio
    async def test_close_cleanup(self, mock_qdrant_client):
        """Test proper cleanup on close."""
        config = QdrantConfig(url="http://localhost:6333")
        store = QdrantStore(qdrant_config=config)
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            assert store._initialized
            
            await store.close()
            assert not store._initialized
            assert not store._collection_ready
            mock_qdrant_client.close.assert_called_once()


@pytest.mark.unit
class TestCollectionManagement:
    """Test collection setup and management."""

    @pytest.mark.asyncio
    async def test_setup_new_collection(self, mock_qdrant_client):
        """Test setting up a new collection."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="new_collection",
            vector_size=768,
            distance_metric="cosine"
        )
        
        # Mock empty collections list (new collection needed)
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])
        
        store = QdrantStore(qdrant_config=config)
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            await store.setup_collection()
            
            # Should create collection
            mock_qdrant_client.create_collection.assert_called_once()
            create_call = mock_qdrant_client.create_collection.call_args
            
            assert create_call[1]['collection_name'] == "new_collection"
            assert create_call[1]['vectors_config'].size == 768

    @pytest.mark.asyncio
    async def test_setup_existing_collection(self, mock_qdrant_client):
        """Test handling existing collection."""
        config = QdrantConfig(url="http://localhost:6333", collection_name="existing")
        
        # Mock existing collection
        existing_collection = MagicMock()
        existing_collection.name = "existing"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )
        
        # Mock collection info for validation
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 1024
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        store = QdrantStore(qdrant_config=config)
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            await store.setup_collection()
            
            # Should not create collection
            mock_qdrant_client.create_collection.assert_not_called()
            # Should validate existing collection
            mock_qdrant_client.get_collection.assert_called_with("existing")

    @pytest.mark.asyncio
    async def test_collection_dimension_mismatch(self, mock_qdrant_client):
        """Test handling of vector dimension mismatch."""
        config = QdrantConfig(
            url="http://localhost:6333",
            vector_size=1024,
            collection_name="test_collection"
        )
        
        # Mock existing collection with different dimension
        existing_collection = MagicMock()
        existing_collection.name = "test_collection"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )
        
        # Mock collection info with wrong dimension
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 512  # Wrong size
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        store = QdrantStore(qdrant_config=config)
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            with pytest.raises(VectorDimensionError, match="dimension mismatch"):
                await store.initialize()


@pytest.mark.unit
class TestBatchOperations:
    """Test batch storage operations."""

    @pytest.fixture
    def sample_repositories(self) -> List[RepositoryData]:
        """Create sample repository data for testing."""
        return [
            RepositoryData(
                repo_name="test/repo1",
                repo_url="https://github.com/test/repo1",
                summary="First test repository",
                tags=["python", "ml"],
                language="Python",
                stars=100,
            ),
            RepositoryData(
                repo_name="test/repo2", 
                repo_url="https://github.com/test/repo2",
                summary="Second test repository",
                tags=["javascript", "web"],
                language="JavaScript",
                stars=50,
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self) -> List[List[float]]:
        """Create sample embeddings for testing."""
        return [
            [0.1] * 1024,  # First embedding
            [0.2] * 1024,  # Second embedding
        ]

    @pytest.mark.asyncio
    async def test_store_repositories_batch_success(
        self, 
        mock_qdrant_client, 
        sample_repositories, 
        sample_embeddings
    ):
        """Test successful batch storage."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        store = QdrantStore(qdrant_config=config)
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            result = await store.store_repositories_batch(
                sample_repositories, 
                sample_embeddings
            )
            
            assert result == 2
            mock_qdrant_client.upsert.assert_called_once()
            
            # Verify statistics updated
            assert store.stats["operations"] == 1
            assert store.stats["points_stored"] == 2

    @pytest.mark.asyncio
    async def test_store_repositories_dimension_mismatch(
        self, 
        mock_qdrant_client, 
        sample_repositories
    ):
        """Test handling of embedding dimension mismatch."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        store = QdrantStore(qdrant_config=config)
        
        # Wrong dimension embeddings
        wrong_embeddings = [[0.1] * 512, [0.2] * 512]  # 512 instead of 1024
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            with pytest.raises(VectorDimensionError, match="doesn't match"):
                await store.store_repositories_batch(
                    sample_repositories, 
                    wrong_embeddings
                )

    @pytest.mark.asyncio
    async def test_store_repositories_count_mismatch(
        self, 
        mock_qdrant_client, 
        sample_repositories, 
        sample_embeddings
    ):
        """Test handling of repository/embedding count mismatch."""
        config = QdrantConfig(url="http://localhost:6333")
        store = QdrantStore(qdrant_config=config)
        
        # Remove one embedding to create mismatch
        mismatched_embeddings = sample_embeddings[:-1]
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            with pytest.raises(ValueError, match="must match number of embeddings"):
                await store.store_repositories_batch(
                    sample_repositories, 
                    mismatched_embeddings
                )

    @pytest.mark.asyncio
    async def test_upsert_retry_on_failure(self, mock_qdrant_client, sample_repositories, sample_embeddings):
        """Test retry logic for upsert failures."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        store = QdrantStore(qdrant_config=config)
        
        # Mock upsert to fail twice, then succeed
        mock_qdrant_client.upsert.side_effect = [
            UnexpectedResponse(status_code=500, reason_phrase="Internal Server Error", content=b"Temporary failure", headers={}),
            UnexpectedResponse(status_code=500, reason_phrase="Internal Server Error", content=b"Another failure", headers={}),
            None  # Success
        ]
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            result = await store.store_repositories_batch(
                sample_repositories, 
                sample_embeddings
            )
            
            assert result == 2
            # Should have called upsert 3 times (2 failures + 1 success)
            assert mock_qdrant_client.upsert.call_count == 3


@pytest.mark.unit
class TestSearchOperations:
    """Test vector search operations."""

    @pytest.mark.asyncio
    async def test_vector_search_success(self, mock_qdrant_client):
        """Test successful vector search."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        store = QdrantStore(qdrant_config=config)
        
        # Mock search results
        mock_hit = MagicMock()
        mock_hit.score = 0.95
        mock_hit.payload = {
            "repo_name": "test/repo",
            "repo_url": "https://github.com/test/repo",
            "summary": "Test repository",
            "tags": ["python", "test"],
            "language": "Python",
            "stars": 100,
        }
        mock_qdrant_client.search.return_value = [mock_hit]
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            query_vector = [0.1] * 1024
            results = await store.vector_search(query_vector, limit=10)
            
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].repo_name == "test/repo"
            assert results[0].score == 0.95
            assert results[0].vector_score == 0.95

    @pytest.mark.asyncio
    async def test_vector_search_dimension_mismatch(self, mock_qdrant_client):
        """Test vector search with wrong dimension."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        store = QdrantStore(qdrant_config=config)
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            # Wrong dimension query vector
            wrong_query_vector = [0.1] * 512  # 512 instead of 1024
            
            with pytest.raises(VectorDimensionError, match="doesn't match"):
                await store.vector_search(wrong_query_vector)

    @pytest.mark.asyncio
    async def test_vector_search_with_filters(self, mock_qdrant_client):
        """Test vector search with filtering conditions."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        store = QdrantStore(qdrant_config=config)
        
        mock_qdrant_client.search.return_value = []
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            query_vector = [0.1] * 1024
            filter_conditions = {
                "tags": ["python", "ml"],
                "language": "Python",
                "min_stars": 100,
            }
            
            await store.vector_search(
                query_vector, 
                filter_conditions=filter_conditions
            )
            
            # Verify search was called with filter
            mock_qdrant_client.search.assert_called_once()
            call_args = mock_qdrant_client.search.call_args
            assert call_args[1]['query_filter'] is not None

    @pytest.mark.asyncio
    async def test_score_threshold_filtering(self, mock_qdrant_client):
        """Test score threshold filtering."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        store = QdrantStore(qdrant_config=config)
        
        # Mock hits with different scores
        mock_hits = []
        for score in [0.95, 0.75, 0.45, 0.25]:
            hit = MagicMock()
            hit.score = score
            hit.payload = {"repo_name": f"repo_{score}", "repo_url": f"url_{score}"}
            mock_hits.append(hit)
        
        mock_qdrant_client.search.return_value = mock_hits
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            query_vector = [0.1] * 1024
            results = await store.vector_search(
                query_vector, 
                score_threshold=0.5
            )
            
            # Should only return results with score >= 0.5
            assert len(results) == 2
            assert all(r.score >= 0.5 for r in results)


@pytest.mark.unit
class TestHealthAndMonitoring:
    """Test health checks and monitoring features."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_qdrant_client):
        """Test successful health check."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        store = QdrantStore(qdrant_config=config)
        
        # Mock collection info and count
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 1024
        mock_collection_info.config.params.vectors.distance = "Cosine"
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        mock_count_result = MagicMock()
        mock_count_result.count = 1000
        mock_qdrant_client.count.return_value = mock_count_result
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            health = await store.health_check()
            
            assert health["status"] == "healthy"
            assert health["points_count"] == 1000
            assert health["vector_size"] == 1024
            assert "response_time_ms" in health
            assert "stats" in health

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_qdrant_client):
        """Test health check with failure."""
        config = QdrantConfig(url="http://localhost:6333")
        store = QdrantStore(qdrant_config=config)
        
        # Mock failure
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            health = await store.health_check() 
            
            assert health["status"] == "unhealthy"
            assert "error" in health
            assert "Collection not found" in health["error"]

    @pytest.mark.asyncio
    async def test_get_existing_repositories(self, mock_qdrant_client):
        """Test getting existing repository names."""
        config = QdrantConfig(url="http://localhost:6333")
        store = QdrantStore(qdrant_config=config)
        
        # Mock scroll responses
        mock_points = [
            MagicMock(payload={"repo_name": "repo1"}),
            MagicMock(payload={"repo_name": "repo2"}),
            MagicMock(payload={"repo_name": "repo3"}),
        ]
        
        mock_qdrant_client.scroll.side_effect = [
            (mock_points[:2], "offset1"),  # First batch
            (mock_points[2:], None),       # Second batch (last)
        ]
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            await store.initialize()
            
            existing = await store.get_existing_repositories()
            
            assert existing == {"repo1", "repo2", "repo3"}
            assert mock_qdrant_client.scroll.call_count == 2

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        config = QdrantConfig(url="http://localhost:6333")
        store = QdrantStore(qdrant_config=config)
        
        # Check initial stats
        assert store.stats["operations"] == 0
        assert store.stats["points_stored"] == 0
        assert store.stats["errors"] == 0
        
        # Test response time update
        store._update_response_time(0.5)
        assert store.stats["avg_response_time"] == 0.5
        
        # Increment operations to enable EMA calculation
        store.stats["operations"] = 1
        store._update_response_time(1.0)
        # Should be exponential moving average: 0.1 * 1.0 + 0.9 * 0.5 = 0.55
        assert 0.5 < store.stats["avg_response_time"] < 1.0


@pytest.mark.integration
class TestQdrantStoreIntegration:
    """Integration tests for QdrantStore with real-like scenarios."""

    @pytest.mark.asyncio
    async def test_full_storage_and_search_cycle(self, mock_qdrant_client):
        """Test complete storage and search cycle."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        
        repositories = [
            RepositoryData(
                repo_name="test/ml-project",
                repo_url="https://github.com/test/ml-project",
                summary="Machine learning project with Python",
                tags=["ml", "python"],
                language="Python",
                stars=500,
            )
        ]
        
        embeddings = [[0.1] * 1024]
        
        # Mock search to return what we stored
        mock_hit = MagicMock()
        mock_hit.score = 0.9
        mock_hit.payload = {
            "repo_name": "test/ml-project",
            "repo_url": "https://github.com/test/ml-project",
            "summary": "Machine learning project with Python",
            "tags": ["ml", "python"],
            "language": "Python",
            "stars": 500,
        }
        mock_qdrant_client.search.return_value = [mock_hit]
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            async with QdrantStore(qdrant_config=config) as store:
                # Store repositories
                stored_count = await store.store_repositories_batch(
                    repositories, embeddings
                )
                assert stored_count == 1
                
                # Search for them
                query_vector = [0.1] * 1024
                results = await store.vector_search(query_vector, limit=10)
                
                assert len(results) == 1
                assert results[0].repo_name == "test/ml-project"
                assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_qdrant_client):
        """Test concurrent storage operations."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024, batch_size=50)
        
        # Create multiple batches
        repositories_batch1 = [
            RepositoryData(
                repo_name=f"test/repo{i}",
                repo_url=f"https://github.com/test/repo{i}",
                summary=f"Repository {i}",
                tags=["test"],
                language="Python",
                stars=i * 10,
            )
            for i in range(25)
        ]
        
        repositories_batch2 = [
            RepositoryData(
                repo_name=f"test/repo{i}",
                repo_url=f"https://github.com/test/repo{i}",
                summary=f"Repository {i}",
                tags=["test"],
                language="JavaScript",
                stars=i * 10,
            )
            for i in range(25, 50)
        ]
        
        embeddings_batch1 = [[0.1] * 1024 for _ in range(25)]
        embeddings_batch2 = [[0.2] * 1024 for _ in range(25)]
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            async with QdrantStore(qdrant_config=config) as store:
                # Run concurrent operations
                results = await asyncio.gather(
                    store.store_repositories_batch(repositories_batch1, embeddings_batch1),
                    store.store_repositories_batch(repositories_batch2, embeddings_batch2),
                )
                
                assert results == [25, 25]
                # Should have called upsert twice
                assert mock_qdrant_client.upsert.call_count == 2

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_batch_performance(self, mock_qdrant_client):
        """Test performance with large batches."""
        config = QdrantConfig(url="http://localhost:6333", vector_size=1024)
        
        # Create large batch
        large_batch_size = 1000
        repositories = [
            RepositoryData(
                repo_name=f"test/repo{i}",
                repo_url=f"https://github.com/test/repo{i}",
                summary=f"Repository {i}" * 10,  # Longer summary
                tags=[f"tag{j}" for j in range(5)],  # Multiple tags
                language="Python",
                stars=i,
            )
            for i in range(large_batch_size)
        ]
        
        embeddings = [[float(i % 100) / 100] * 1024 for i in range(large_batch_size)]
        
        with patch('src.core.storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            async with QdrantStore(qdrant_config=config) as store:
                start_time = time.time()
                
                result = await store.store_repositories_batch(repositories, embeddings)
                
                end_time = time.time()
                duration = end_time - start_time
                
                assert result == large_batch_size
                # Should complete within reasonable time (mocked, so very fast)
                assert duration < 5.0
                
                # Verify statistics
                assert store.stats["points_stored"] == large_batch_size