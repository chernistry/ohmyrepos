"""Enterprise-grade vector storage with comprehensive batch processing and security."""

import asyncio
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import QdrantConfig, settings
from src.core.logging import LoggerMixin, PerformanceLogger, log_exception
from src.core.models import RepositoryData, SearchResult


class StorageError(Exception):
    """Base exception for storage operations."""

    def __init__(self, message: str, operation: str = "") -> None:
        super().__init__(message)
        self.operation = operation


class VectorDimensionError(StorageError):
    """Vector dimension mismatch error."""

    pass


class ConnectionError(StorageError):
    """Qdrant connection error."""

    pass


class QdrantStore(LoggerMixin):
    """Enterprise-grade Qdrant vector store with comprehensive features.
    
    Features:
    - Async-first design with proper resource management
    - Batch processing with configurable sizes
    - Comprehensive error handling and retry logic
    - Performance monitoring and metrics
    - Security hardening and validation
    - Connection pooling and health checks
    """

    def __init__(
        self,
        qdrant_config: Optional[QdrantConfig] = None,
        collection_name: str = "repositories",
        batch_size: int = 100,
        max_concurrent_batches: int = 3,
    ) -> None:
        """Initialize the Qdrant store.

        Args:
            qdrant_config: Qdrant configuration
            collection_name: Name of the collection
            batch_size: Batch size for operations
            max_concurrent_batches: Maximum concurrent batch operations
        """
        super().__init__()
        
        if qdrant_config:
            self.config = qdrant_config
        elif settings.qdrant:
            self.config = settings.qdrant
        else:
            raise ValueError("Qdrant configuration is required")
        
        # Use config's collection_name if provided, otherwise use parameter
        self.collection_name = self.config.collection_name or collection_name
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        
        # Initialize client and state
        self._client: Optional[AsyncQdrantClient] = None
        self._initialized = False
        self._collection_ready = False
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Statistics
        self.stats = {
            "operations": 0,
            "points_stored": 0,
            "points_searched": 0,
            "errors": 0,
            "avg_response_time": 0.0,
        }

    async def __aenter__(self) -> "QdrantStore":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize the Qdrant client and validate connection."""
        if self._initialized:
            return

        try:
            # Initialize async client
            client_kwargs = {"url": str(self.config.url)}
            
            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key.get_secret_value()
            
            # Use sync client first to verify connection
            try:
                from qdrant_client import QdrantClient
                sync_client = QdrantClient(url=str(self.config.url))
                sync_client.get_collections()
                self.logger.debug("Successfully connected to Qdrant using sync client")
            except Exception as sync_error:
                self.logger.error(f"Failed to connect using sync client: {sync_error}")
                raise ConnectionError(f"Failed to connect to Qdrant: {sync_error}")
            
            # Now initialize async client
            try:
                self._client = AsyncQdrantClient(**client_kwargs)
                self.logger.debug(f"Initialized Qdrant client with URL: {self.config.url}")
            except Exception as client_error:
                raise ConnectionError(f"Failed to create Qdrant client: {client_error}")
            
            # Ensure client is properly initialized
            if not self._client:
                raise ConnectionError("Failed to initialize Qdrant client")
                
            # Validate connection
            await self._validate_connection()
            
            # Setup collection
            await self.setup_collection()
            
            self._initialized = True
            self.logger.info("Qdrant store initialized successfully")
            
        except VectorDimensionError:
            # Re-raise VectorDimensionError as-is
            await self.close()
            raise
        except Exception as e:
            await self.close()
            raise ConnectionError(f"Failed to initialize Qdrant store: {e}")

    async def _validate_connection(self) -> None:
        """Validate Qdrant connection and permissions."""
        try:
            # Ensure client is initialized before attempting connection
            if not self._client:
                raise ConnectionError("Qdrant client is not initialized")
                
            # Test connection with a simple operation
            collections = await self._client.get_collections()
            if not collections:
                raise ConnectionError("Received empty response from Qdrant")
                
            self.logger.debug(f"Found {len(collections.collections)} collections")
            
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Qdrant: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((UnexpectedResponse, ConnectionError)),
    )
    async def setup_collection(self) -> None:
        """Setup collection with proper configuration and indexes."""
        if self._collection_ready:
            return

        try:
            with PerformanceLogger(self.logger, "setup_collection"):
                # Verify client is initialized
                if not self._client:
                    raise StorageError("Cannot setup collection: Qdrant client is not initialized")
                
                # Check if collection exists
                try:
                    collections = await self._client.get_collections()
                    if not collections:
                        raise StorageError("Failed to get collections from Qdrant")
                    
                    collection_names = [c.name for c in collections.collections]
                except Exception as e:
                    raise StorageError(f"Failed to list collections: {e}")

                if self.collection_name not in collection_names:
                    self.logger.info(f"Creating collection: {self.collection_name}")
                    
                    # Create collection with optimized settings
                    await self._client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=self.config.vector_size,
                            distance=getattr(
                                qdrant_models.Distance, 
                                self.config.distance_metric.upper()
                            ),
                        ),
                        optimizers_config=qdrant_models.OptimizersConfig(
                            default_segment_number=2,
                            max_segment_size=20000,
                            deleted_threshold=0.2,
                            vacuum_min_vector_number=1000,
                            flush_interval_sec=5,
                        ),
                        hnsw_config=qdrant_models.HnswConfig(
                            m=16,
                            ef_construct=100,
                            full_scan_threshold=10000,
                        ),
                    )

                    # Create payload indexes for efficient filtering
                    await self._create_payload_indexes()
                    
                else:
                    # Validate existing collection
                    await self._validate_collection()

                self._collection_ready = True
                self.logger.info(f"Collection {self.collection_name} is ready")

        except VectorDimensionError:
            # Re-raise VectorDimensionError as-is
            raise
        except Exception as e:
            log_exception(self.logger, e, "Failed to setup collection")
            raise StorageError(f"Collection setup failed: {e}")

    async def _create_payload_indexes(self) -> None:
        """Create optimized payload indexes."""
        indexes = [
            ("tags", qdrant_models.PayloadSchemaType.KEYWORD),
            ("language", qdrant_models.PayloadSchemaType.KEYWORD),
            ("stars", qdrant_models.PayloadSchemaType.INTEGER),
            ("repo_name", qdrant_models.PayloadSchemaType.TEXT),
        ]

        for field_name, field_type in indexes:
            try:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                self.logger.debug(f"Created index for field: {field_name}")
            except Exception as e:
                self.logger.warning(f"Failed to create index for {field_name}: {e}")

    async def _validate_collection(self) -> None:
        """Validate existing collection configuration."""
        try:
            collection_info = await self._client.get_collection(self.collection_name)
            existing_size = collection_info.config.params.vectors.size
            
            if existing_size != self.config.vector_size:
                raise VectorDimensionError(
                    f"Collection dimension mismatch: expected {self.config.vector_size}, "
                    f"found {existing_size}"
                )
                
        except VectorDimensionError:
            # Re-raise VectorDimensionError as-is
            raise
        except Exception as e:
            if "not found" in str(e).lower():
                # Collection was deleted, recreate
                await self.setup_collection()
            else:
                raise StorageError(f"Collection validation failed: {e}")

    async def store_repositories_batch(
        self,
        repositories: List[RepositoryData],
        embeddings: List[List[float]],
    ) -> int:
        """Store repositories with their embeddings in batch.

        Args:
            repositories: List of repository data
            embeddings: Corresponding embeddings

        Returns:
            Number of repositories stored

        Raises:
            StorageError: If storage operation fails
        """
        if not self._initialized:
            await self.initialize()

        if len(repositories) != len(embeddings):
            raise ValueError("Number of repositories must match number of embeddings")

        start_time = time.time()
        
        try:
            async with self._semaphore:
                points = []
                
                for repo, embedding in zip(repositories, embeddings):
                    # Generate deterministic but secure ID
                    repo_id = self._generate_secure_id(repo.repo_name)
                    
                    # Validate embedding dimension
                    if len(embedding) != self.config.vector_size:
                        raise VectorDimensionError(
                            f"Embedding dimension {len(embedding)} doesn't match "
                            f"collection dimension {self.config.vector_size}"
                        )
                    
                    # Prepare payload with validation
                    payload = self._prepare_payload(repo)
                    
                    # Create point
                    point = qdrant_models.PointStruct(
                        id=repo_id,
                        vector=embedding,
                        payload=payload,
                    )
                    points.append(point)

                # Batch upsert with retry
                await self._upsert_points_with_retry(points)
                
                # Update statistics
                self.stats["operations"] += 1
                self.stats["points_stored"] += len(points)
                self._update_response_time(time.time() - start_time)
                
                self.logger.info(f"Stored {len(points)} repositories in batch")
                return len(points)

        except VectorDimensionError:
            # Re-raise VectorDimensionError as-is
            self.stats["errors"] += 1
            raise
        except Exception as e:
            self.stats["errors"] += 1
            log_exception(
                self.logger, e, "Failed to store repositories batch", 
                batch_size=len(repositories)
            )
            raise StorageError(f"Batch storage failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((UnexpectedResponse, ConnectionError)),
    )
    async def _upsert_points_with_retry(
        self, points: List[qdrant_models.PointStruct]
    ) -> None:
        """Upsert points with retry logic."""
        await self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def _generate_secure_id(self, repo_name: str) -> str:
        """Generate a secure, deterministic ID for a repository."""
        # Use SHA-256 with salt for security
        salt = f"ohmyrepos_v2_{self.collection_name}"
        name_hash = hashlib.sha256(f"{salt}:{repo_name}".encode()).hexdigest()
        return str(uuid.UUID(name_hash[:32]))

    def _prepare_payload(self, repo: RepositoryData) -> Dict[str, Any]:
        """Prepare and validate payload for storage."""
        # Sanitize and validate payload data
        payload = {
            "repo_name": repo.repo_name[:200],  # Limit length
            "repo_url": repo.repo_url[:500],
            "summary": repo.summary[:2000] if repo.summary else "",
            "description": repo.description[:1000] if repo.description else "",
            "tags": repo.tags[:20],  # Limit number of tags
            "language": repo.language[:50] if repo.language else None,
            "stars": max(0, repo.stars),  # Ensure non-negative
            "forks": max(0, repo.forks),
            "created_at": repo.created_at,
            "updated_at": repo.updated_at,
            "indexed_at": time.time(),
        }

        # Remove None values
        return {k: v for k, v in payload.items() if v is not None}

    async def vector_search(
        self,
        query_vector: List[float],
        limit: int = 25,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Perform vector similarity search.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_conditions: Optional filters
            score_threshold: Minimum similarity score

        Returns:
            List of search results

        Raises:
            StorageError: If search operation fails
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        
        try:
            # Validate query vector
            if len(query_vector) != self.config.vector_size:
                raise VectorDimensionError(
                    f"Query vector dimension {len(query_vector)} doesn't match "
                    f"collection dimension {self.config.vector_size}"
                )

            # Build filter
            query_filter = self._build_filter(filter_conditions)

            # Perform search
            results = await self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

            # Convert to SearchResult objects
            search_results = []
            for hit in results:
                if hit.score >= score_threshold:
                    result = SearchResult(
                        repo_name=hit.payload.get("repo_name", ""),
                        repo_url=hit.payload.get("repo_url", ""),
                        summary=hit.payload.get("summary"),
                        tags=hit.payload.get("tags", []),
                        language=hit.payload.get("language"),
                        stars=hit.payload.get("stars", 0),
                        score=hit.score,
                        vector_score=hit.score,
                    )
                    search_results.append(result)

            # Update statistics
            self.stats["operations"] += 1
            self.stats["points_searched"] += len(search_results)
            self._update_response_time(time.time() - start_time)

            self.logger.debug(
                f"Vector search completed: {len(search_results)} results, "
                f"avg_score={sum(r.score for r in search_results) / len(search_results) if search_results else 0:.3f}"
            )

            return search_results

        except VectorDimensionError:
            # Re-raise VectorDimensionError as-is
            self.stats["errors"] += 1
            raise
        except Exception as e:
            self.stats["errors"] += 1
            log_exception(self.logger, e, "Vector search failed")
            raise StorageError(f"Vector search failed: {e}")

    def _build_filter(
        self, conditions: Optional[Dict[str, Any]]
    ) -> Optional[qdrant_models.Filter]:
        """Build Qdrant filter from conditions."""
        if not conditions:
            return None

        must_conditions = []

        # Tag filtering
        if "tags" in conditions:
            tags = conditions["tags"]
            if isinstance(tags, list) and tags:
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key="tags",
                        match=qdrant_models.MatchAny(any=tags),
                    )
                )

        # Language filtering
        if "language" in conditions and conditions["language"]:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="language",
                    match=qdrant_models.MatchValue(value=conditions["language"]),
                )
            )

        # Stars range filtering
        if "min_stars" in conditions or "max_stars" in conditions:
            stars_range = {}
            if "min_stars" in conditions:
                stars_range["gte"] = conditions["min_stars"]
            if "max_stars" in conditions:
                stars_range["lte"] = conditions["max_stars"]
            
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="stars",
                    range=qdrant_models.Range(**stars_range),
                )
            )

        return qdrant_models.Filter(must=must_conditions) if must_conditions else None

    async def get_existing_repositories(self) -> Set[str]:
        """Get set of repository names that already exist in the collection.

        Returns:
            Set of existing repository names
        """
        if not self._initialized:
            await self.initialize()

        try:
            existing_repos = set()
            offset = None
            
            while True:
                # Scroll through points
                response = await self._client.scroll(
                    collection_name=self.collection_name,
                    offset=offset,
                    limit=1000,
                    with_vectors=False,
                    with_payload=True,
                )

                points, next_offset = response

                # Extract repo names
                for point in points:
                    if "repo_name" in point.payload:
                        existing_repos.add(point.payload["repo_name"])

                # Check if done
                if not next_offset or len(points) == 0:
                    break
                    
                offset = next_offset

            self.logger.info(f"Found {len(existing_repos)} existing repositories")
            return existing_repos

        except Exception as e:
            log_exception(self.logger, e, "Failed to get existing repositories")
            return set()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the Qdrant store.

        Returns:
            Health check results
        """
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()

            # Check collection status
            collection_info = await self._client.get_collection(self.collection_name)
            
            # Count points
            count_result = await self._client.count(collection_name=self.collection_name)
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "collection_name": self.collection_name,
                "points_count": count_result.count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "stats": self.stats.copy(),
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "unhealthy",
                "response_time_ms": response_time,
                "error": str(e),
                "stats": self.stats.copy(),
            }

    def _update_response_time(self, duration: float) -> None:
        """Update average response time statistics."""
        if self.stats["operations"] == 0:
            self.stats["avg_response_time"] = duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats["avg_response_time"] = (
                alpha * duration + (1 - alpha) * self.stats["avg_response_time"]
            )

    async def close(self) -> None:
        """Close the Qdrant client and clean up resources."""
        if self._client:
            await self._client.close()
            self._client = None
        
        self._initialized = False
        self._collection_ready = False
        
        self.logger.info("Qdrant store closed")


@asynccontextmanager
async def qdrant_store(
    qdrant_config: Optional[QdrantConfig] = None,
    **kwargs: Any,
) -> AsyncGenerator[QdrantStore, None]:
    """Context manager for Qdrant store.
    
    Args:
        qdrant_config: Qdrant configuration
        **kwargs: Additional store arguments
        
    Yields:
        Initialized QdrantStore instance
    """
    store = QdrantStore(qdrant_config=qdrant_config, **kwargs)
    try:
        async with store:
            yield store
    finally:
        await store.close()