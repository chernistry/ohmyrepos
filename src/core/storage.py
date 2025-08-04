"""Vector storage for Oh My Repos.

This module provides functionality to store and retrieve repository embeddings.
"""

import logging
import uuid
import hashlib
from typing import Dict, List, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

# Fix imports for compatibility
try:
    from src.config import settings
    from src.core.embeddings import EmbeddingFactory, EmbeddingProvider
except ImportError:
    from config import settings
    from core.embeddings import EmbeddingFactory, EmbeddingProvider

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class VectorDimensionError(StorageError):
    """Vector dimension mismatch error."""

    pass


# Constants
COLLECTION_NAME = "repositories"
BATCH_SIZE = 100


class QdrantStore:
    """Qdrant vector store for repository embeddings.

    This class handles storing and retrieving repository embeddings in Qdrant.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        """Initialize the Qdrant store.

        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            embedding_provider: Embedding provider to use
        """
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY

        if not self.url:
            raise ValueError("Qdrant URL must be provided")

        # Initialize client with correct parameters
        if self.api_key:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
        else:
            self.client = QdrantClient(url=self.url)

        self.embedding_provider = embedding_provider or EmbeddingFactory.get_provider()
        logger.debug(f"Initialized QdrantStore with URL: {self.url}")

    async def setup_collection(self) -> None:
        """Set up the collection for storing repository embeddings.

        This method creates the collection if it doesn't exist.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            # Get embedding dimension from provider
            embedding_dimension = self.embedding_provider.dimension
            logger.info(f"Using embedding dimension: {embedding_dimension}")

            if COLLECTION_NAME not in collection_names:
                logger.info(f"Creating collection: {COLLECTION_NAME}")

                # Create collection
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=embedding_dimension,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )

                # Create payload index for filtering
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="tags",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                )
            else:
                # Check dimension of existing collection
                collection_info = self.client.get_collection(COLLECTION_NAME)
                existing_dimension = collection_info.config.params.vectors.size

                if existing_dimension != embedding_dimension:
                    logger.warning(
                        f"Collection dimension mismatch: expected {embedding_dimension}, "
                        f"but collection has {existing_dimension}. "
                        f"Recreating collection."
                    )

                    # Delete and recreate collection with correct dimension
                    self.client.delete_collection(COLLECTION_NAME)

                    self.client.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=qdrant_models.VectorParams(
                            size=embedding_dimension,
                            distance=qdrant_models.Distance.COSINE,
                        ),
                    )

                    # Create payload index for filtering
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name="tags",
                        field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                    )

            logger.info(f"Collection {COLLECTION_NAME} is ready")

        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=0.1),
        retry=retry_if_exception_type((UnexpectedResponse,)),
        reraise=True,
    )
    async def get_existing_repo_ids(self) -> List[str]:
        """Get IDs of repositories that already have embeddings.

        Returns:
            List of repository IDs that are already in the vector database

        Raises:
            StorageError: When unable to retrieve existing repository IDs
        """
        try:
            # Check if collection exists first
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if COLLECTION_NAME not in collection_names:
                logger.info(f"Collection {COLLECTION_NAME} does not exist yet")
                return []

            # Get count of points in collection
            count_response = self.client.count(
                collection_name=COLLECTION_NAME, count_filter=None
            )

            if not hasattr(count_response, "count") or count_response.count == 0:
                logger.info(f"Collection {COLLECTION_NAME} is empty")
                return []

            # Get all points (with scroll API if needed)
            all_points = []
            offset = None
            limit = 100  # Fetch in batches
            max_iterations = 1000  # Safety limit to prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                try:
                    points_response = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        offset=offset,
                        limit=limit,
                        with_vectors=False,  # We don't need vectors, just IDs
                        with_payload=True,  # We need payload to get repo names
                    )

                    if not points_response or len(points_response) < 2:
                        logger.warning("Invalid scroll response format")
                        break

                    points, next_offset = points_response[0], points_response[1]

                    # Extract repo names from payload
                    for point in points:
                        if (
                            hasattr(point, "payload")
                            and point.payload
                            and "repo_name" in point.payload
                        ):
                            repo_name = point.payload.get("repo_name", "")
                            if repo_name and isinstance(repo_name, str):
                                all_points.append(repo_name)

                    # Check if we've reached the end
                    if len(points) < limit or not next_offset:
                        break

                    offset = next_offset
                    iteration += 1

                except Exception as e:
                    logger.error(f"Error during scroll iteration {iteration}: {e}")
                    if iteration == 0:
                        # If first iteration fails, re-raise
                        raise
                    # Otherwise, return what we have so far
                    break

            if iteration >= max_iterations:
                logger.warning(
                    f"Reached maximum iterations ({max_iterations}) while scrolling"
                )

            logger.info(
                f"Found {len(all_points)} repositories already stored in vector database"
            )
            return all_points

        except UnexpectedResponse as e:
            logger.error(f"Qdrant API error getting existing repository IDs: {e}")
            raise StorageError(f"Failed to retrieve existing repositories: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting existing repository IDs: {e}")
            raise StorageError(f"Unexpected error retrieving repositories: {e}") from e

    async def store_repositories(
        self, repositories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Store repository embeddings in Qdrant.

        Args:
            repositories: List of repository data dictionaries

        Returns:
            The updated repositories list with embedding status marked
        """
        if not repositories:
            logger.warning("No repositories to store")
            return repositories

        logger.info(f"Processing {len(repositories)} repositories for storage")

        # Get existing repository IDs
        existing_repo_ids = await self.get_existing_repo_ids()
        logger.info(f"Found {len(existing_repo_ids)} repositories already in Qdrant")

        # Filter out repositories that already have embeddings
        to_process = []
        for repo in repositories:
            repo_name = repo.get("repo_name", "") or repo.get("name", "")

            if repo_name in existing_repo_ids:
                # Mark as already having embeddings
                repo["has_embedding"] = True
                logger.debug(f"Repository {repo_name} already has embeddings")
            else:
                # Mark for processing
                to_process.append(repo)
                repo["has_embedding"] = False

        if not to_process:
            logger.info("All repositories already have embeddings, nothing to store")
            return repositories

        logger.info(f"Storing {len(to_process)} new repositories in Qdrant")

        # Process in batches
        for i in range(0, len(to_process), BATCH_SIZE):
            batch = to_process[i : i + BATCH_SIZE]
            await self._store_batch(batch)

            # Mark processed repositories
            for repo in batch:
                repo["has_embedding"] = True

                # Also update status in the original repositories list
                repo_name = repo.get("name", "")
                for original_repo in repositories:
                    if original_repo.get("name", "") == repo_name:
                        original_repo["has_embedding"] = True
                        break

        # Check that all repositories have the has_embedding flag
        repos_with_embeddings = sum(
            1 for repo in repositories if repo.get("has_embedding", False)
        )
        logger.info(
            f"Total repositories with embeddings: {repos_with_embeddings} out of {len(repositories)}"
        )

        logger.info("Repository storage complete")
        return repositories

    async def _store_batch(self, repositories: List[Dict[str, Any]]) -> None:
        """Store a batch of repositories.

        Args:
            repositories: Batch of repository data
        """
        # Extract summaries for embedding
        summaries = [
            f"{repo.get('repo_name', '')} - {repo.get('summary', '')}"
            for repo in repositories
        ]

        # Generate embeddings
        embeddings = await self.embedding_provider.embed_documents(summaries)

        # Check vector dimension
        if embeddings:
            logger.info(
                f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}"
            )

        # Prepare points for Qdrant
        points = []
        for i, repo in enumerate(repositories):
            # Generate a secure UUID based on the repo name and a salt
            repo_name = repo.get("name", "")
            if not repo_name:
                logger.warning(f"Repository at index {i} has no name, skipping")
                continue

            # Use SHA-256 with a salt for security (deterministic but secure)
            salt = "ohmyrepos_v1"  # Version salt to allow migration if needed
            name_hash = hashlib.sha256(f"{salt}:{repo_name}".encode()).hexdigest()
            repo_id = str(uuid.UUID(name_hash[:32]))

            # Prepare payload
            payload = {
                "repo_name": repo.get("repo_name", ""),
                "repo_url": repo.get("repo_url", ""),
                "summary": repo.get("summary", ""),
                "tags": repo.get("tags", []),
                "language": repo.get("language"),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
            }

            # Create point
            point = qdrant_models.PointStruct(
                id=repo_id,
                vector=embeddings[i],
                payload=payload,
            )

            points.append(point)

        # Upsert points to Qdrant
        try:
            # Get collection info to check dimension
            collection_info = self.client.get_collection(COLLECTION_NAME)
            expected_dimension = collection_info.config.params.vectors.size
            actual_dimension = len(embeddings[0]) if embeddings else 0

            if expected_dimension != actual_dimension:
                logger.error(
                    f"Vector dimension mismatch: collection expects {expected_dimension}, but got {actual_dimension}"
                )
                raise ValueError(
                    f"Vector dimension mismatch: collection expects {expected_dimension}, but got {actual_dimension}"
                )

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            logger.info(f"Stored {len(points)} repositories")

        except UnexpectedResponse as e:
            logger.error(f"Error storing repositories: {e}")
            raise

    async def search(
        self,
        query: str,
        limit: int = 10,
        filter_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for repositories by query.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_tags: Optional list of tags to filter by

        Returns:
            List of matching repository data
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_query(query)

        # Prepare filter if tags provided
        filter_query = None
        if filter_tags:
            filter_query = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="tags",
                        match=qdrant_models.MatchAny(any=filter_tags),
                    )
                ]
            )

        # Search in Qdrant
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_query,
            )

            # Extract and return results
            return [
                {
                    "repo_name": hit.payload.get("repo_name", ""),
                    "repo_url": hit.payload.get("repo_url", ""),
                    "summary": hit.payload.get("summary", ""),
                    "tags": hit.payload.get("tags", []),
                    "language": hit.payload.get("language"),
                    "stars": hit.payload.get("stars", 0),
                    "score": hit.score,
                }
                for hit in results
            ]

        except Exception as e:
            logger.error(f"Error searching repositories: {e}")
            return []

    async def close(self) -> None:
        """Close connections."""
        await self.embedding_provider.close()
