"""Ollama embedding provider implementation."""

import logging
from typing import List, Optional, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider."""

    def __init__(self, config: Any) -> None:
        """Initialize Ollama embedding provider.

        Args:
            config: Embedding configuration (Pydantic model or EmbeddingConfig object)
        """
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=str(config.base_url),
            timeout=config.timeout,
            follow_redirects=True,
        )
        # Initialize dimension if provided in config, otherwise it will be detected
        self._dimension = getattr(config, "dimension", None)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self._embed_single(text)
            embeddings.append(embedding)
        
        # Update dimension if not set
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
            logger.info(f"Detected embedding dimension: {self._dimension}")

        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        embedding = await self._embed_single(text)
        
        # Update dimension if not set
        if self._dimension is None:
            self._dimension = len(embedding)
            logger.info(f"Detected embedding dimension: {self._dimension}")
            
        return embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = await self.client.post(
                "",
                json={
                    "model": self.config.model,
                    "prompt": text,
                },
            )
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            logger.error(f"Failed to generate embedding with Ollama: {e}")
            raise

    async def close(self) -> None:
        """Close the provider resources."""
        await self.client.aclose()

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        if self._dimension is None:
            # If dimension is not known yet, we might need to fetch it or default
            # For now, return 0 or raise error, but better to have it detected.
            # Let's default to 0 and let the caller handle or wait for first embed.
            return 0
        return self._dimension
