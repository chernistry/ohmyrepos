"""Jina AI embedding provider.

This module provides an embedding provider using Jina AI's embedding models.
"""

import logging
from typing import List, Optional
import time

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from src.config import settings
from src.core.embeddings.base import EmbeddingProvider, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding operations."""

    pass


class EmbeddingServiceUnavailable(EmbeddingError):
    """Embedding service is temporarily unavailable."""

    pass


class JinaEmbeddingProvider(EmbeddingProvider):
    """Jina AI embedding provider.

    This class provides embeddings using Jina AI's embedding models.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
    ) -> None:
        """Initialize the Jina AI embedding provider.

        Args:
            config: Configuration for the provider
        """
        self.config = config or EmbeddingConfig(
            model_name=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_MODEL_API_KEY,
            api_url=settings.EMBEDDING_MODEL_URL,
        )

        self.api_url = self.config.api_url or "https://api.jina.ai/v1/embeddings"
        self.api_key = self.config.api_key
        self.model_name = self.config.model_name

        if not self.api_key:
            raise ValueError("Jina AI API key is required")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Configure connection pooling and timeouts for production
        limits = httpx.Limits(
            max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
        )

        timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

        self.client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            http2=True,  # Enable HTTP/2 for better performance
        )

        # Set default dimension based on model
        if self.config.dimensions is not None:
            self._dimension = self.config.dimensions
        elif "v3" in self.model_name:
            self._dimension = 1024  # jina-embeddings-v3 uses 1024 dimensions
        else:
            self._dimension = 768  # jina-embeddings-v2 uses 768 dimensions

        logger.debug(
            f"Initialized JinaEmbeddingProvider with model: {self.model_name} (dimension: {self._dimension})"
        )

    async def embed_query(self, text: str) -> List[float]:
        """Embed a query text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        embeddings = await self.embed_documents([text])
        return embeddings[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=0.1),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with retry and error handling.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: When embedding generation fails
            EmbeddingServiceUnavailable: When service is temporarily unavailable
        """
        if not texts:
            return []

        # Validate input
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Text at index {i} must be a string, got {type(text)}"
                )
            if len(text.strip()) == 0:
                raise ValueError(f"Text at index {i} cannot be empty")
            if len(text) > 8192:  # Jina's max input length
                logger.warning(
                    f"Text at index {i} truncated from {len(text)} to 8192 characters"
                )
                texts[i] = text[:8192]

        # Prepare payload
        payload = {
            "model": self.model_name,
            "input": texts,
        }

        start_time = time.time()
        logger.debug(
            f"Generating embeddings for {len(texts)} texts using {self.model_name}"
        )

        try:
            response = await self.client.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()

            result = response.json()

            # Validate response structure
            if "data" not in result:
                raise ValueError("Invalid response format: missing 'data' field")

            if len(result["data"]) != len(texts):
                raise ValueError(
                    f"Response count mismatch: expected {len(texts)}, got {len(result['data'])}"
                )

            embeddings = []
            for i, item in enumerate(result["data"]):
                if "embedding" not in item:
                    raise ValueError(f"Missing embedding for item {i}")
                embeddings.append(item["embedding"])

            # Update dimension if not set
            if self.config.dimensions is None and embeddings:
                self._dimension = len(embeddings[0])
                logger.info(f"Detected embedding dimension: {self._dimension}")

            # Validate dimensions for all vectors
            for i, emb in enumerate(embeddings):
                if len(emb) != self._dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self._dimension}, got {len(emb)} for text {i}"
                    )

            duration = time.time() - start_time
            logger.debug(f"Generated {len(embeddings)} embeddings in {duration:.2f}s")

            return embeddings

        except httpx.TimeoutException as e:
            logger.error(
                f"Timeout generating embeddings after {time.time() - start_time:.2f}s: {e}"
            )
            raise EmbeddingServiceUnavailable(f"Request timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error generating embeddings: {e.response.status_code} - {e.response.text}"
            )
            if e.response.status_code >= 500:
                raise EmbeddingServiceUnavailable(
                    f"Server error: {e.response.status_code}"
                ) from e
            elif e.response.status_code == 429:
                raise EmbeddingServiceUnavailable("Rate limit exceeded") from e
            else:
                raise EmbeddingError(
                    f"Client error: {e.response.status_code} - {e.response.text}"
                ) from e
        except httpx.ConnectError as e:
            logger.error(f"Connection error generating embeddings: {e}")
            raise EmbeddingServiceUnavailable(f"Connection failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}")
            raise EmbeddingError(f"Unexpected error: {e}") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension
